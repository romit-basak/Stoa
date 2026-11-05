"""Scoring for flow models: the numbers every model is compared and selected on.

Every flow model is scored here, the same way, on out-of-fold predictions: each sensor's
hours predicted by the fold model that held that sensor out (the splits come from
src/features/training_frame.py). The rules are fixed before any model is trained:

- Score only the hours after each sensor's calibration window (`scored_mask`), so calibrated
  and uncalibrated predictions are measured on the same hours.
- Every model gets WAPE, log MAE, MAE in people, RMSE, overall bias and how well it ranks
  sensors by volume (`score`). Which one decides, and by how much a model must win, is set
  with the thresholds, which the team hasn't decided. Log MAE is the proposal: it weighs a
  quiet and a busy street alike (0.5 means off by about 1.6 times either way).
- Each location fold is scored on its own and the folds are averaged (`cv_metrics`), under
  the names MLflow runs use (docs/models/experiment_tracking.md): `wape_fold0`, `wape_cv`.
- Event days, other days and each era are also scored on their own (`standard_parts`).
- The holdout precincts are scored once, for the final model only, with the same `score`;
  `cv_metrics` refuses their rows.
- A model's 80% interval is scored by how many counts fall inside it (`cv_coverage`), and
  per-site calibration by the fold scores after scaling each sensor to its first weeks of
  counts (`calibrated_cv_metrics`, names ending in `_cal`).

Per-slice errors for the bias report come from src.bias.slice_errors.slice_errors, given the
same predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.bias.slice_errors import bias_share, calibrate, calibration_window, mae, wape
from src.bias.slices import era


def scored_mask(
    sensors: np.ndarray, ts_utc: pd.Series, calibration_days: float
) -> np.ndarray:
    """True for the hours every model is scored on: those after the calibration window."""
    return ~calibration_window(sensors, ts_utc, calibration_days)


def log_mae(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error of log(1 + count): the proposed deciding score."""
    return float(np.abs(np.log1p(pred) - np.log1p(y)).mean())


def score(
    rows: pd.DataFrame, y: np.ndarray, pred: np.ndarray
) -> dict[str, float | int]:
    """Headline scores for one set of predictions; `rows` needs `location_id`, row for row."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if not len(rows) == len(y) == len(pred):
        raise ValueError(
            f"{len(rows)} rows, {len(y)} counts and {len(pred)} predictions"
        )
    if len(y) == 0:
        raise ValueError("nothing to score")
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError("predictions must be finite and non-negative")

    per = pd.DataFrame(
        {"s": rows["location_id"].to_numpy(), "y": y, "p": pred, "e": np.abs(pred - y)}
    ).groupby("s")
    totals = per[["y", "e"]].sum()
    sensor_wape = (totals["e"] / totals["y"])[totals["y"] > 0]
    # Averages, not totals: busy sensors with long histories would rank first on totals alone.
    means = per[["y", "p"]].mean()
    ranked = len(means) >= 3 and means["y"].nunique() > 1 and means["p"].nunique() > 1
    return {
        "log_mae": log_mae(y, pred),
        "wape": wape(y, pred),
        "mae": mae(y, pred),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "bias": bias_share(y, pred),
        "sensor_rank_corr": float(spearmanr(means["y"], means["p"]).statistic)
        if ranked
        else float("nan"),
        "sensor_wape_median": float(sensor_wape.median()),
        "sensor_wape_worst": float(sensor_wape.max()),
        "rows": len(y),
        "sensors": len(means),
    }


CV_METRICS = ("wape", "log_mae", "mae", "rmse", "bias", "sensor_rank_corr")


def cv_metrics(
    rows: pd.DataFrame, y: np.ndarray, pred: np.ndarray, folds: np.ndarray
) -> dict[str, float]:
    """Each location fold's scores and their mean, named for MLflow: `wape_fold0`, `wape_cv`.

    `folds` is each row's fold from splits.parquet. Holdout rows (fold -1) are refused: the
    holdout precincts never take part in choosing a model (CLAUDE.md core rule 2).
    """
    y, pred, folds = (np.asarray(y, float), np.asarray(pred, float), np.asarray(folds))
    if len(folds) != len(rows) or len(folds) == 0:
        raise ValueError(f"{len(rows)} rows and {len(folds)} folds")
    if (folds < 0).any():
        raise ValueError("holdout rows (fold -1) can't be scored with the folds")
    out: dict[str, float] = {}
    per_fold: dict[str, list[float]] = {name: [] for name in CV_METRICS}
    for k in np.unique(folds):
        m = folds == k
        s = score(rows[m], y[m], pred[m])
        for name in CV_METRICS:
            out[f"{name}_fold{int(k)}"] = s[name]
            per_fold[name].append(s[name])
    out.update({f"{name}_cv": float(np.nanmean(v)) for name, v in per_fold.items()})
    return out


def standard_parts(
    rows: pd.DataFrame, event_dates: pd.Series, eras: dict[str, str]
) -> dict[str, np.ndarray]:
    """Masks for the parts scored on their own: event days, other days, and each era."""
    dates = pd.to_datetime(rows["date_local"])
    on_event = dates.isin(pd.to_datetime(event_dates)).to_numpy()
    parts = {"event_days": on_event, "other_days": ~on_event}
    labels = era(dates, eras).to_numpy()
    for name in eras:
        parts[f"era_{name}"] = labels == name
    return parts


def score_parts(
    rows: pd.DataFrame, y: np.ndarray, pred: np.ndarray, parts: dict[str, np.ndarray]
) -> dict[str, dict[str, float | int]]:
    """`score` on each part; an empty part reports 0 rows instead of failing."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return {
        name: score(rows[m], y[m], pred[m]) if m.any() else {"rows": 0, "sensors": 0}
        for name, m in parts.items()
    }


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Share of counts inside [low, high]: about 0.8 for good 10% and 90% quantiles."""
    y, low, high = (np.asarray(a, dtype=float) for a in (y, low, high))
    if not len(y) == len(low) == len(high):
        raise ValueError(
            f"{len(y)} counts, {len(low)} lower and {len(high)} upper bounds"
        )
    if len(y) == 0:
        raise ValueError("nothing to score")
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("interval bounds must be finite")
    return float(((y >= low) & (y <= high)).mean())


def cv_coverage(
    y: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    folds: np.ndarray,
    name: str = "coverage80",
) -> dict[str, float]:
    """Each location fold's interval coverage and their mean: `coverage80_fold0`, `_cv`.

    Holdout rows (fold -1) are refused, as in `cv_metrics`.
    """
    y, low, high, folds = (
        np.asarray(y, float),
        np.asarray(low, float),
        np.asarray(high, float),
        np.asarray(folds),
    )
    if len(folds) != len(y) or len(folds) == 0:
        raise ValueError(f"{len(y)} counts and {len(folds)} folds")
    if (folds < 0).any():
        raise ValueError("holdout rows (fold -1) can't be scored with the folds")
    out: dict[str, float] = {}
    for k in np.unique(folds):
        m = folds == k
        out[f"{name}_fold{int(k)}"] = coverage(y[m], low[m], high[m])
    out[f"{name}_cv"] = float(np.mean(list(out.values())))
    return out


def calibrated_cv_metrics(
    rows: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    folds: np.ndarray,
    calibration_days: float,
) -> dict[str, float]:
    """`cv_metrics` after per-site calibration, each name ending in `_cal`: `log_mae_cv_cal`.

    Each sensor's predictions are scaled so they add up to its counts in its first
    calibration_days (CLAUDE.md, "shape vs. scale"), then scored on the hours after that
    window: the hours `scored_mask` keeps for the uncalibrated scores. `rows` needs
    `location_id` and `ts_utc`, and covers the window hours too.
    """
    y, pred, folds = (np.asarray(y, float), np.asarray(pred, float), np.asarray(folds))
    if not len(rows) == len(y) == len(pred) == len(folds):
        raise ValueError(
            f"{len(rows)} rows, {len(y)} counts, {len(pred)} predictions "
            f"and {len(folds)} folds"
        )
    sensors = rows["location_id"].to_numpy()
    window = calibration_window(sensors, rows["ts_utc"], calibration_days)
    cal = calibrate(pred, y, sensors, window)
    keep = ~window
    m = cv_metrics(rows[keep], y[keep], cal[keep], folds[keep])
    return {f"{k}_cal": v for k, v in m.items()}
