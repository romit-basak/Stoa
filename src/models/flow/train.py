"""Train the flow model's candidates and the baseline ladder on the location folds.

Usage: python -m src.models.flow.train [--variant rich|portable|both] [--families ...]
       [--tune] [--n-trials N] [--folds N] [--max-trees N] [--train-rows N]
       [--output-dir DIR]

For each variant, every candidate family (src/models/flow/candidates.py) trains on each fold's
training sample and predicts that fold's sensors, so every prediction comes from a model that
never saw the sensor or its neighbours (CLAUDE.md core rule 2). The baseline ladder
(src/models/flow/baselines.py) is fitted on the same plan. Holdout precincts are in no fold;
src.models.flow.select scores them for the selected model only.

Each model is one MLflow run in experiment stoa-flow (src/models/tracking.py records the code
and data version): `<family>_<variant>` for candidates, `baseline_<rung>` for the ladder, with
the fold metrics named as in docs/models/experiment_tracking.md. The baselines use only network
features, which both variants share, so they run once.

Writes to <output_dir>/<variant>/: oof.parquet (one prediction column per model) and
candidates.json (each model's run id, parameters and metrics). src.models.flow.select reads
them next.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.features.common import log
from src.ingest.common import setup_logging
from src.models import tracking
from src.models.flow import baselines, candidates
from src.models.flow import data as flow_data
from src.models.flow import tune as flow_tune
from src.models.flow.data import Data
from src.models.flow.metrics import cv_metrics


def plan_tag(n_folds: int, train_rows: int, seed: int) -> str:
    """Identifies a fold plan, so the gate only compares runs scored on the same rows."""
    return f"folds={n_folds};train_rows={train_rows};seed={seed}"


def oof_candidates(
    data: Data,
    families: list[str],
    params: dict[str, dict[str, Any]],
    plan: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Out-of-fold predictions per family, building each fold's matrices once for all."""
    n = len(data.y)
    preds = {f: np.full(n, np.nan, dtype="float32") for f in families}
    seconds = dict.fromkeys(families, 0.0)
    for k, (train, test) in enumerate(plan):
        x_train, x_test = data.frame.matrix(train), data.frame.matrix(test)
        for f in families:
            t = time.perf_counter()
            model = candidates.make(f, params[f], seed).fit(x_train, data.y[train])
            preds[f][test] = candidates.predict(model, x_test)
            seconds[f] += time.perf_counter() - t
            log.info("fold %d %s: %.0f s", k, f, time.perf_counter() - t)
        del x_train, x_test
    return preds, seconds


def oof_baselines(
    data: Data, plan: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, np.ndarray]:
    return {
        f"baseline_{r}": baselines.oof(
            r, data.frame.hourly, data.frame.osm, data.y, plan
        )
        for r in baselines.RUNGS
    }


def log_run(
    name: str,
    params: dict[str, Any],
    tags: dict[str, str],
    metrics: dict[str, float],
) -> str:
    with tracking.run(name, params=params, tags=tags) as active:
        mlflow.log_metrics(metrics)
        return active.info.run_id


def train_variant(
    data: Data,
    fcfg: dict[str, Any],
    families: list[str],
    plan: list[tuple[np.ndarray, np.ndarray]],
    out: Path,
    params: dict[str, dict[str, Any]],
    baseline_preds: dict[str, np.ndarray],
    baseline_runs: dict[str, dict[str, Any]],
    plan_id: str,
) -> dict[str, Any]:
    """Candidates for one variant, logged; writes oof.parquet and candidates.json."""
    preds, seconds = oof_candidates(data, families, params, plan, fcfg["seed"])
    models: dict[str, Any] = {}
    for f in families:
        metrics = cv_metrics(
            data.y, preds[f], plan, data.sensors, data.ts, fcfg["calibration_days"]
        )
        run_id = log_run(
            f"{f}_{data.variant}",
            {**params[f], "variant": data.variant, "family": f},
            {
                "stoa.role": "candidate",
                "stoa.family": f,
                "stoa.variant": data.variant,
                "stoa.plan": plan_id,
            },
            {**metrics, "fit_s": seconds[f]},
        )
        models[f] = {"run_id": run_id, "params": params[f], "metrics": metrics}
        log.info(
            "%s_%s: mae_log_cv %.3f, wape_cv %.3f",
            f,
            data.variant,
            metrics["mae_log_cv"],
            metrics["wape_cv"],
        )

    out.mkdir(parents=True, exist_ok=True)
    tested = np.concatenate([t for _, t in plan])
    oof = pd.DataFrame(
        {
            "row": tested,
            "location_id": data.sensors[tested],
            "ts_utc": data.ts.to_numpy()[tested],
            "fold": data.fold[tested],
            "y": data.y[tested].astype("float32"),
        }
    )
    for name, p in {**preds, **baseline_preds}.items():
        oof[name] = p[tested]
    oof.to_parquet(out / "oof.parquet", index=False)
    summary = {
        "variant": data.variant,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan": plan_id,
        "n_folds": len(plan),
        "train_rows": int(fcfg["train_rows_per_fold"]),
        "seed": int(fcfg["seed"]),
        "families": families,
        "models": models,
        "baselines": baseline_runs,
    }
    (out / "candidates.json").write_text(json.dumps(summary, indent=2))
    return summary


def run(
    variants: list[str],
    families: list[str],
    fcfg: dict[str, Any],
    out_root: Path,
    load_data=flow_data.load,
    n_folds: int | None = None,
    max_trees: int | None = None,
    tune: bool = False,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Train every variant; returns each variant's candidates.json content."""
    seed, train_rows = fcfg["seed"], fcfg["train_rows_per_fold"]
    baseline_preds: dict[str, np.ndarray] | None = None
    baseline_runs: dict[str, dict[str, Any]] = {}
    reference: np.ndarray | None = None
    summaries = {}
    for variant in variants:
        t = time.perf_counter()
        data = load_data(variant)
        plan = flow_data.fold_plan(data.fold, train_rows, seed, n_folds)
        plan_id = plan_tag(len(plan), train_rows, seed)
        params = {
            f: candidates.capped(f, fcfg["candidates"][f]["params"], max_trees)
            for f in families
        }
        if tune:
            for f in families:
                tags = {
                    "stoa.role": "tuning",
                    "stoa.family": f,
                    "stoa.variant": variant,
                }
                with tracking.run(f"tune_{f}_{variant}", tags=tags):
                    best = flow_tune.tune(data, f, fcfg, n_trials, max_trees)
                params[f] = candidates.capped(f, best, max_trees)

        # Both variants have the same rows; the baselines use network features only.
        if baseline_preds is None:
            reference = data.sensors
            baseline_preds = oof_baselines(data, plan)
            for name, p in baseline_preds.items():
                metrics = cv_metrics(
                    data.y, p, plan, data.sensors, data.ts, fcfg["calibration_days"]
                )
                bparams = {
                    "rung": name.removeprefix("baseline_"),
                    "ridge_alpha": "leave-one-sensor-out",
                }
                run_id = log_run(
                    name,
                    bparams,
                    {
                        "stoa.role": "baseline",
                        "stoa.family": name,
                        "stoa.variant": "network",
                        "stoa.plan": plan_id,
                    },
                    metrics,
                )
                baseline_runs[name] = {"run_id": run_id, "metrics": metrics}
                log.info("%s: mae_log_cv %.3f", name, metrics["mae_log_cv"])
        elif not np.array_equal(reference, data.sensors):
            raise ValueError("the variants' training frames have different rows")

        summaries[variant] = train_variant(
            data,
            fcfg,
            families,
            plan,
            out_root / variant,
            params,
            baseline_preds,
            baseline_runs,
            plan_id,
        )
        log.info("%s: %.0f s", variant, time.perf_counter() - t)
        del data
    return summaries


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=["rich", "portable", "both"], default="both")
    ap.add_argument("--families", nargs="+", choices=candidates.FAMILIES)
    ap.add_argument(
        "--tune", action="store_true", help="Optuna search per family first"
    )
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--folds", type=int, help="use only the first N folds (quick runs)")
    ap.add_argument("--max-trees", type=int, help="cap trees per model (quick runs)")
    ap.add_argument("--train-rows", type=int, help="training rows per fold")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    fcfg = flow_data.flow_config()
    if args.train_rows:
        fcfg["train_rows_per_fold"] = args.train_rows
    out = flow_data.output_dir(fcfg, args.output_dir)
    variants = fcfg["variants"] if args.variant == "both" else [args.variant]
    families = args.families or list(fcfg["candidates"])
    run(
        variants,
        families,
        fcfg,
        out,
        n_folds=args.folds,
        max_trees=args.max_trees,
        tune=args.tune,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
