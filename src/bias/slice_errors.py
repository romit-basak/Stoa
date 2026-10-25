"""Per-slice model error on held-out sensors, and mitigations for it (bias steps 2-3).

Usage: python -m src.bias.slice_errors [--if-older-than DAYS]

Training takes about 3 minutes per mitigation, so the DAG runs it with --if-older-than 7: it
skips itself (successfully) when the last run is under a week old and used the same settings.

Trains a baseline (scikit-learn gradient boosting, Poisson loss) with whole sensors held out
(grouped K-fold, CLAUDE.md core rule 2), so every located sensor gets predictions from a model
that never saw it. Each slice from src/bias/slices.py then gets its error, measured with
Fairlearn's MetricFrame. The run is repeated once per mitigation in configs/bias.yaml
(`errors.mitigations`), so the report shows what each one does to every slice and to the
overall error. A mitigation either reweights the training rows (and retrains) or calibrates
the baseline's predictions to each sensor's first weeks of counts. Those first weeks are kept
out of scoring for every mitigation, so all are compared on the same hours.

Writes to data/processed/bias/melbourne/:
- slice_errors.csv: one row per (mitigation, dimension, group);
- mitigation_summary.csv: one row per mitigation (overall error and the worst groups);
- slice_errors_meta.json: features, folds, model settings.

The baseline stands in for the real flow model, which doesn't exist yet; the same functions
take its out-of-fold predictions later (step 4). Its features follow core rule 9: no sensor
or segment IDs, no coordinates, no month or day of year. Slices are only for checking; none
is a model feature.

Error is WAPE (sum of absolute errors / sum of counts), so busy groups don't dominate and
quiet ones (night) count. Confidence intervals resample whole sensors, because hours from one
sensor aren't independent.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.ensemble import HistGradientBoostingRegressor

from src.bias.representation import network_reference, sliced_rows
from src.bias.slices import load_bias_config
from src.features.common import config_hash, load_features_config, log, write_meta
from src.features.training_frame import (
    add_derived,
    derived_features,
    design_matrix,
    feature_names,
    fold_plan,
    row_features,
    sensor_features,
)
from src.ingest.common import REPO_ROOT, load_config, setup_logging

# The baseline's features are the training frame's: the hourly table's own features, then the
# matched segment's (src/features/training_frame.py, chosen by role in src/features/schema.py).
ROW_FEATURES = row_features()
# Shade and events, which the frame derives per row (training_frame.add_derived).
DERIVED_FEATURES = derived_features()


# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# Weights (mitigations)
# --------------------------------------------------------------------------------------------


def weights_for(
    spec: dict[str, Any],
    sample: pd.DataFrame,
    references: dict[str, pd.Series],
) -> np.ndarray | None:
    """Training-row weights for one mitigation, scaled to mean 1 (None = unweighted).

    - none: every row counts the same, so long-running busy sensors dominate;
    - sensor_equal: every sensor's rows sum to the same weight;
    - group_equal: every group of `dimension` sums to the same weight;
    - reference: each group of `dimension` is weighted to its share of a reference (for
      precinct, the walkable network), so over-sampled precincts count less.
    """
    kind = spec.get("weight", "none")
    if kind == "none":
        return None
    if kind == "sensor_equal":
        key = sample["location_id"]
    elif kind in ("group_equal", "reference"):
        key = sample[spec["dimension"]].astype(str)
    else:
        raise ValueError(f"unknown weight {kind!r}")
    counts = key.map(key.value_counts()).to_numpy(dtype=float)
    w = 1.0 / counts
    if kind == "reference":
        ref = references[spec["dimension"]]
        ref = ref / ref.sum()
        share = key.map(ref).to_numpy(dtype=float)
        # Groups without a reference keep their sample share.
        sample_share = counts / len(key)
        w = np.where(np.isnan(share), sample_share, share) / counts
    return w / w.mean()


# --------------------------------------------------------------------------------------------
# Calibration (mitigation by post-processing)
# --------------------------------------------------------------------------------------------


def calibration_window(sensors: np.ndarray, ts: pd.Series, days: float) -> np.ndarray:
    """True for each sensor's first `days` days of rows: the counts used to calibrate it.

    Every mitigation is scored on the other rows only, so calibrated and uncalibrated
    errors are measured on the same hours.
    """
    first = pd.Series(ts.to_numpy()).groupby(sensors).transform("min")
    return (ts.to_numpy() - first.to_numpy()) < pd.Timedelta(days=days)


def calibrate(
    pred: np.ndarray, y: np.ndarray, sensors: np.ndarray, window: np.ndarray
) -> np.ndarray:
    """Scale each sensor's predictions so they add up to its counts in the window.

    This is the per-site calibration of CLAUDE.md ("shape vs. scale"): the model gives the
    pattern, a few weeks of local counts give the volume. A sensor with no predicted
    pedestrians in its window keeps its predictions.
    """
    frame = pd.DataFrame({"s": sensors, "y": np.where(window, y, 0.0)})
    frame["p"] = np.where(window, pred, 0.0)
    sums = frame.groupby("s")[["y", "p"]].sum()
    scale = (sums["y"] / sums["p"]).where(sums["p"] > 0, 1.0)
    return pred * scale.reindex(sensors).to_numpy()


# --------------------------------------------------------------------------------------------
# Out-of-fold predictions
# --------------------------------------------------------------------------------------------


def oof_predict(
    build_x: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
    weights: Callable[[np.ndarray], np.ndarray | None],
    model_params: dict[str, Any],
) -> np.ndarray:
    """Predictions for every eligible row from the fold model that held its sensor out."""
    pred = np.full(len(y), np.nan)
    for k, (train, test) in enumerate(plan):
        t = time.perf_counter()
        model = HistGradientBoostingRegressor(loss="poisson", **model_params)
        model.fit(build_x(train), y[train], sample_weight=weights(train))
        for chunk in np.array_split(test, max(1, len(test) // 500_000)):
            pred[chunk] = model.predict(build_x(chunk))
        log.info(
            "fold %d: trained on %d rows, predicted %d in %.0f s",
            k,
            len(train),
            len(test),
            time.perf_counter() - t,
        )
    return pred


# --------------------------------------------------------------------------------------------
# Metrics per slice
# --------------------------------------------------------------------------------------------


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).sum() / y_true.sum())


def bias_share(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Over (+) or under (-) prediction, as a share of the true total."""
    return float((y_pred - y_true).sum() / y_true.sum())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).mean())


def cell_sums(
    groups: pd.Series, sensors: pd.Series, y: np.ndarray, pred: np.ndarray
) -> pd.DataFrame:
    """Totals per (group, sensor) cell: count, prediction, absolute error and rows.

    WAPE, bias and MAE are ratios of sums, so they can be computed exactly from these cells,
    which is what makes MetricFrame and the bootstrap fast on millions of rows.
    """
    frame = pd.DataFrame(
        {
            "group": groups.to_numpy(),
            "sensor": sensors.to_numpy(),
            "y": y,
            "pred": pred,
            "abs_err": np.abs(pred - y),
            "n": 1,
        }
    )
    return frame.groupby(["group", "sensor"], observed=True).sum().reset_index()


def _wape(y_true, y_pred, abs_err, n):
    return float(np.sum(abs_err) / np.sum(y_true))


def _bias(y_true, y_pred, abs_err, n):
    return float((np.sum(y_pred) - np.sum(y_true)) / np.sum(y_true))


def _mae(y_true, y_pred, abs_err, n):
    return float(np.sum(abs_err) / np.sum(n))


def cell_metrics(cells: pd.DataFrame) -> MetricFrame:
    """Fairlearn MetricFrame of WAPE, bias and MAE per group, from (group, sensor) cells."""
    params = {"abs_err": cells["abs_err"].to_numpy(), "n": cells["n"].to_numpy()}
    return MetricFrame(
        metrics={"wape": _wape, "bias": _bias, "mae": _mae},
        y_true=cells["y"].to_numpy(),
        y_pred=cells["pred"].to_numpy(),
        sensitive_features=cells["group"].astype(str).rename("group"),
        sample_params={"wape": params, "bias": params, "mae": params},
    )


def sensor_bootstrap(cells: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """95% interval of each group's WAPE, resampling the group's sensors with replacement."""
    rng = np.random.default_rng(seed)
    out = {}
    for g, part in cells.groupby("group", observed=True):
        e, t = part["abs_err"].to_numpy(), part["y"].to_numpy()
        pick = rng.integers(0, len(e), size=(reps, len(e)))
        boot = e[pick].sum(axis=1) / t[pick].sum(axis=1)
        out[str(g)] = (np.nanpercentile(boot, 2.5), np.nanpercentile(boot, 97.5))
    return pd.DataFrame(out, index=["wape_lo", "wape_hi"]).T


def slice_errors(
    rows: pd.DataFrame,
    dims: list[str],
    y: np.ndarray,
    pred: np.ndarray,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Error per group for every slice, with sensor-bootstrap intervals."""
    ecfg = cfg["errors"]
    overall = wape(y, pred)
    parts = []
    for dim in dims:
        cells = cell_sums(rows[dim], rows["location_id"], y, pred)
        mf = cell_metrics(cells)
        t = mf.by_group.copy()
        by_group = cells.assign(group=cells["group"].astype(str)).groupby("group")
        t["n_rows"] = by_group["n"].sum()
        t["n_sensors"] = by_group["sensor"].nunique()
        t = t.join(sensor_bootstrap(cells, ecfg["bootstrap_reps"], ecfg["seed"]))
        t["wape_vs_overall"] = t["wape"] / overall
        # Worse: the whole 95% interval is above the overall error (by the margin, if any).
        t["worse"] = t["wape_lo"] > overall * (1 + ecfg["worse_margin"])
        t["disparity_ratio"] = mf.ratio(method="between_groups")["wape"]
        t = t.reset_index(names="group")
        t.insert(0, "dimension", dim)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def summarize_mitigation(
    name: str, errors: pd.DataFrame, y: np.ndarray, pred: np.ndarray
) -> dict[str, Any]:
    worst = errors.sort_values("wape", ascending=False).drop_duplicates("dimension")
    return {
        "mitigation": name,
        "wape": wape(y, pred),
        "bias": bias_share(y, pred),
        "mae": mae(y, pred),
        "groups_worse": int(errors["worse"].sum()),
        "worse_groups": "; ".join(
            f"{d}={g}"
            for d, g in errors.loc[errors["worse"], ["dimension", "group"]].values
        ),
        "max_wape_group": "; ".join(
            f"{d}={g} ({w:.2f})"
            for d, g, w in worst[["dimension", "group", "wape"]].head(5).values
        ),
    }


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def build(
    data_cfg: dict[str, Any], fcfg: dict[str, Any], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ecfg = cfg["errors"]
    site = cfg["site"]
    feats = REPO_ROOT / fcfg["processed_dir"] / "features" / site
    extra = ROW_FEATURES + ["ts_local", "sun_azimuth_deg"]
    sliced = sliced_rows(data_cfg, fcfg, cfg, extra_columns=extra)
    hourly, rows = sliced["hourly"], sliced["rows"]
    hourly = add_derived(hourly, site, data_cfg, fcfg)
    row_cols = ROW_FEATURES + DERIVED_FEATURES
    # Step 2 measures error, so the slices that only describe data coverage are left out:
    # every evaluated sensor is located, and its source and era are already sliced.
    dims = [d for d in sliced["dims"] if d not in ecfg["skip_dimensions"]]

    osm = sensor_features(pd.read_parquet(feats / "sensor_osm_features.parquet"))
    eligible = np.flatnonzero(
        (rows["has_location"] == "located").to_numpy() & ~rows["outage"].to_numpy()
    )
    y_all = hourly["pedestriancount"].to_numpy(dtype=float)
    y = y_all[eligible]
    groups = hourly["location_id"].to_numpy()[eligible]
    cap = fcfg["built_form"]["max_distance_m"]

    def build_x(pos: np.ndarray) -> np.ndarray:
        return design_matrix(hourly, eligible[pos], osm, cap, row_cols)

    plan = fold_plan(groups, ecfg["folds"], ecfg["train_rows_per_fold"], ecfg["seed"])
    edges = gpd.read_parquet(
        REPO_ROOT / fcfg["interim_dir"] / "osm" / site / "graph" / "edges.parquet",
        columns=["length_m", "geometry"],
    )
    references = {"precinct": network_reference(edges, sliced["blocks"], cfg)}
    eval_rows = rows.iloc[eligible].reset_index(drop=True)
    window = calibration_window(
        groups, hourly["ts_utc"].iloc[eligible], ecfg["calibration_days"]
    )
    scored = ~window
    scored_rows = eval_rows[scored].reset_index(drop=True)

    preds: dict[str, np.ndarray] = {}
    all_errors, summaries = [], []
    for spec in ecfg["mitigations"]:
        name = spec["name"]
        t = time.perf_counter()

        def weights(pos: np.ndarray, spec: dict[str, Any] = spec) -> np.ndarray | None:
            return weights_for(spec, eval_rows.iloc[pos], references)

        if "calibrate_from" in spec:
            pred = calibrate(preds[spec["calibrate_from"]], y, groups, window)
        else:
            pred = oof_predict(build_x, y, plan, weights, ecfg["model"])
        preds[name] = pred
        errors = slice_errors(scored_rows, dims, y[scored], pred[scored], cfg)
        errors.insert(0, "mitigation", name)
        all_errors.append(errors)
        summaries.append(summarize_mitigation(name, errors, y[scored], pred[scored]))
        log.info(
            "%s: WAPE %.3f, %d groups worse than overall, %.0f s",
            name,
            summaries[-1]["wape"],
            summaries[-1]["groups_worse"],
            time.perf_counter() - t,
        )

    meta = {
        "rows_evaluated": int(scored.sum()),
        "rows_in_calibration_windows": int(window.sum()),
        "sensors_evaluated": len(np.unique(groups)),
        "folds": ecfg["folds"],
        "train_rows_per_fold": ecfg["train_rows_per_fold"],
        "model": {
            "estimator": "HistGradientBoostingRegressor",
            "loss": "poisson",
            **ecfg["model"],
        },
        "features": feature_names(osm, row_cols),
        "dimensions": dims,
    }
    return pd.concat(all_errors, ignore_index=True), pd.DataFrame(summaries), meta


def is_recent(meta_path: Path, settings_hash: str, days: float, now: datetime) -> bool:
    """True when the last run is younger than `days` and used the same settings."""
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    if meta.get("config_hash") != settings_hash or "run_at" not in meta:
        return False
    return now - datetime.fromisoformat(meta["run_at"]) < timedelta(days=days)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--if-older-than",
        type=float,
        metavar="DAYS",
        help="skip when the last run is younger than DAYS and used the same settings",
    )
    args = ap.parse_args()
    data_cfg = load_config()
    fcfg = load_features_config()
    cfg = load_bias_config()
    out = REPO_ROOT / fcfg["processed_dir"] / "bias" / cfg["site"]
    settings = config_hash({"errors": cfg["errors"], "slices": cfg})
    now = datetime.now(UTC)
    if args.if_older_than is not None and is_recent(
        out / "slice_errors_meta.json", settings, args.if_older_than, now
    ):
        log.info("slice errors are under %g days old; skipped", args.if_older_than)
        return
    t = time.perf_counter()
    errors, summary, meta = build(data_cfg, fcfg, cfg)
    out.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out / "slice_errors.csv", index=False, float_format="%.4f")
    summary.to_csv(out / "mitigation_summary.csv", index=False, float_format="%.4f")
    write_meta(
        out / "slice_errors_meta.json",
        config_hash=settings,
        run_at=now.isoformat(),
        runtime_s=round(time.perf_counter() - t, 1),
        **meta,
    )
    log.info("wrote %s", out / "slice_errors.csv")


if __name__ == "__main__":
    main()
