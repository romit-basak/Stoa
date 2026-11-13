"""Bias mitigation for the selected flow model: reweight its training rows, or calibrate it.

Usage: python -m src.models.flow.mitigate [--variant rich|portable|both] [--output-dir DIR]

Runs after src.models.flow.select. For each mitigation in `flow.mitigation`
(configs/models.yaml), the selected family is retrained with its selected parameters on the
same location folds as training, with the training rows weighted by src/bias/slice_errors.py's
`weights_for`, or (`calibrate_from`) an earlier mitigation's predictions are scaled to each
sensor's first `calibration_days` of counts. Every mitigation is then scored like selection
scores the candidates: the fold metrics, and WAPE per slice of configs/bias.yaml (Fairlearn
MetricFrame with sensor-bootstrap intervals), all on the hours after each sensor's calibration
window, so calibrated and uncalibrated errors cover the same hours.

Each mitigation is one MLflow run, `mitigation_<name>_<variant>` with stoa.role = mitigation.
Writes <output_dir>/<variant>/mitigation/: summary.csv (one row per mitigation), slices.csv
(every group's WAPE under every mitigation) and mitigation.png. The findings and trade-offs
are written up in docs/bias/model_bias.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.bias.slice_errors import calibrate, slice_errors, weights_for
from src.features.common import log
from src.ingest.common import setup_logging
from src.models import tracking
from src.models.flow import candidates, select
from src.models.flow import data as flow_data
from src.models.flow.data import Data, calibration_window
from src.models.flow.evaluate import scored_mask
from src.models.flow.train import fold_scores, plan_tag

# Groups whose WAPE the summary lists for every mitigation, as dimension=group.
WATCHED = ("volume_tier=quiet", "time_of_day=night", "era=covid")


def retrain(
    data: Data,
    family: str,
    params: dict[str, Any],
    plan: list[tuple[np.ndarray, np.ndarray]],
    rows: pd.DataFrame,
    spec: dict[str, Any],
    seed: int,
) -> np.ndarray:
    """Out-of-fold predictions of `family` with each fold's training rows weighted by `spec`."""
    pred = np.full(len(data.y), np.nan, dtype="float32")
    for k, (train, test) in enumerate(plan):
        w = weights_for(spec, rows.iloc[train].reset_index(drop=True), {})
        model = candidates.fit(
            candidates.make(family, params, seed),
            data.frame.matrix(train),
            data.y[train],
            sample_weight=w,
        )
        pred[test] = candidates.predict(model, data.frame.matrix(test))
        log.info("%s fold %d done", spec["name"], k)
    return pred


def calibrated(
    data: Data, pred: np.ndarray, tested: np.ndarray, days: float
) -> np.ndarray:
    """`pred` with each tested sensor scaled to its counts in its first `days` days."""
    out = np.full(len(data.y), np.nan, dtype="float32")
    sensors = data.sensors[tested]
    window = calibration_window(sensors, data.ts.iloc[tested], days)
    out[tested] = calibrate(pred[tested].astype(float), data.y[tested], sensors, window)
    return out


def score(
    data: Data,
    pred: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
    rows: pd.DataFrame,
    dims: list[str],
    bias_cfg: dict[str, Any],
    days: float,
) -> tuple[dict[str, float], pd.DataFrame, tuple[float, str]]:
    """Fold metrics, per-slice errors and the worst slice of out-of-fold predictions."""
    tested = np.concatenate([t for _, t in plan])
    keys = data.frame.hourly.iloc[tested][["location_id", "ts_utc"]]
    keep = scored_mask(keys["location_id"].to_numpy(), keys["ts_utc"], days)
    idx = tested[keep]
    errors = slice_errors(
        rows.iloc[idx].reset_index(drop=True),
        dims,
        data.y[idx].astype(float),
        pred[idx].astype(float),
        bias_cfg,
    )
    metrics = fold_scores(data, pred, plan, days)
    return metrics, errors, select.worst_slice(errors, bias_cfg["thin_min_sensors"])


def group_wape(errors: pd.DataFrame, name: str) -> float:
    dim, group = name.split("=", 1)
    hit = errors[(errors["dimension"] == dim) & (errors["group"].astype(str) == group)]
    return float(hit["wape"].iloc[0]) if len(hit) else float("nan")


def chart(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    names = summary["mitigation"]
    a.barh(names, summary["wape_cv"], color="#2e86c1")
    a.set_xlabel("WAPE, mean over location folds (lower is better)")
    b.barh(names, summary["worst_slice_ratio"], color="#c0392b")
    b.axvline(1.0, color="#555", linewidth=0.8)
    b.set_xlabel("Worst slice WAPE / overall WAPE (lower is fairer)")
    for ax in (a, b):
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mitigate_variant(
    data: Data, fcfg: dict[str, Any], out: Path, slice_fn=select.slice_rows
) -> pd.DataFrame:
    summary = json.loads((out / "candidates.json").read_text())
    family = json.loads((out / "selected.json").read_text())["family"]
    params = summary["models"][family]["params"]
    seed, days = fcfg["seed"], fcfg["calibration_days"]
    plan = flow_data.summary_plan(data, summary)
    oof = pd.read_parquet(out / "oof.parquet", columns=["row", family])
    tested = oof["row"].to_numpy()

    keys = data.frame.hourly[["location_id", "ts_utc"]].reset_index(drop=True)
    rows, dims, bias_cfg = slice_fn(keys)
    dims = [d for d in dims if d not in fcfg["selection"]["skip_dimensions"]]

    preds: dict[str, np.ndarray] = {}
    results, tables = [], []
    for spec in fcfg["mitigation"]:
        name = spec["name"]
        if "calibrate_from" in spec:
            preds[name] = calibrated(data, preds[spec["calibrate_from"]], tested, days)
        elif spec.get("weight", "none") == "none":
            p = np.full(len(data.y), np.nan, dtype="float32")
            p[tested] = oof[family].to_numpy()
            preds[name] = p
        else:
            preds[name] = retrain(data, family, params, plan, rows, spec, seed)
        metrics, errors, (ratio, worst) = score(
            data, preds[name], plan, rows, dims, bias_cfg, days
        )
        errors.insert(0, "mitigation", name)
        tables.append(errors)
        row = {
            "mitigation": name,
            "log_mae_cv": metrics["log_mae_cv"],
            "wape_cv": metrics["wape_cv"],
            "bias_cv": metrics["bias_cv"],
            "worst_slice_ratio": ratio,
            "worst_slice": worst,
            "groups_worse": int(errors["worse"].sum()),
            **{f"wape {g}": group_wape(errors, g) for g in WATCHED},
        }
        results.append(row)
        log.info(
            "%s %s: wape_cv %.3f, worst slice %.2f (%s), %d groups worse",
            data.variant,
            name,
            row["wape_cv"],
            ratio,
            worst,
            row["groups_worse"],
        )
        tags = {
            "stoa.role": "mitigation",
            "stoa.family": family,
            "stoa.variant": data.variant,
            "stoa.mitigation": name,
            "stoa.plan": plan_tag(
                len(plan),
                summary["train_rows"],
                summary["seed"],
                summary.get("train_start"),
            ),
        }
        run_params = {**params, "family": family, "variant": data.variant, **spec}
        with tracking.run(
            f"mitigation_{name}_{data.variant}", params=run_params, tags=tags
        ):
            mlflow.log_metrics(
                {
                    **metrics,
                    "worst_slice_ratio": ratio,
                    "groups_worse": row["groups_worse"],
                }
            )
            mlflow.log_text(errors.to_csv(index=False), "slices.csv")

    table = pd.DataFrame(results)
    art = out / "mitigation"
    art.mkdir(parents=True, exist_ok=True)
    table.to_csv(art / "summary.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(art / "slices.csv", index=False)
    chart(table, art / "mitigation.png")
    return table


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=["rich", "portable", "both"], default="both")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    fcfg = flow_data.flow_config()
    out = flow_data.output_dir(fcfg, args.output_dir)
    variants = fcfg["variants"] if args.variant == "both" else [args.variant]
    for v in variants:
        table = mitigate_variant(flow_data.load(v), fcfg, out / v)
        print(f"\n{v}\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
