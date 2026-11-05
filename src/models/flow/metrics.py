"""Error metrics for the flow model, named as in docs/models/experiment_tracking.md.

`*_cv` metrics are the mean over the location folds. mae_log (MAE on log1p counts) is the
headline success criterion; WAPE (sum of absolute errors over sum of counts) is what the bias
slicing uses, so quiet hours still count without busy sensors dominating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.flow.data import calibrate, calibration_window


def wape(y: np.ndarray, pred: np.ndarray) -> float:
    total = float(np.sum(y))
    return float(np.sum(np.abs(pred - y)) / total) if total > 0 else float("nan")


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def mae_log(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.log1p(y) - np.log1p(np.maximum(pred, 0.0)))))


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean((y >= low) & (y <= high)))


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "wape": wape(y, pred),
        "mae": mae(y, pred),
        "rmse": rmse(y, pred),
        "mae_log": mae_log(y, pred),
    }


def cv_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
    sensors: np.ndarray,
    ts: pd.Series,
    calibration_days: float,
) -> dict[str, float]:
    """Mean over folds of each score, WAPE per fold, and the calibrated scores.

    Calibrated: each held-out sensor's predictions scaled to its first calibration_days of
    counts, then scored on its later hours only (the per-site calibration of CLAUDE.md).
    """
    per_fold = []
    out: dict[str, float] = {}
    for k, (_, test) in enumerate(plan):
        s = scores(y[test], pred[test])
        per_fold.append(s)
        out[f"wape_fold{k}"] = s["wape"]
    for name in per_fold[0]:
        out[f"{name}_cv"] = float(np.mean([s[name] for s in per_fold]))

    test = np.concatenate([t for _, t in plan])
    window = calibration_window(sensors[test], ts.iloc[test], calibration_days)
    cal = calibrate(pred[test], y[test], sensors[test], window)
    later = ~window
    out["wape_cv_cal"] = wape(y[test][later], cal[later])
    out["mae_log_cv_cal"] = mae_log(y[test][later], cal[later])
    return out


def cv_coverage(
    y: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    return float(np.mean([coverage(y[t], low[t], high[t]) for _, t in plan]))


def calibration_curve(
    y: np.ndarray,
    pred: np.ndarray,
    sensors: np.ndarray,
    ts: pd.Series,
    calibration_days: float,
    ks: list[int],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Error on the other sensors after scaling the model to k sensors' first weeks of counts.

    One scale for the whole site, from the k sensors' calibration windows, applied to every
    other sensor and scored on their hours after the window. k = 0 is the uncalibrated model.
    Rows are out-of-fold, so no sensor is scored by a model that trained on it.
    """
    rng = np.random.default_rng(seed)
    window = calibration_window(sensors, ts, calibration_days)
    ids = np.unique(sensors)
    rows = []
    for k in [0, *ks]:
        if k >= len(ids):
            continue
        for rep in range(reps if k else 1):
            chosen = rng.choice(ids, size=k, replace=False) if k else ids[:0]
            fit = np.isin(sensors, chosen) & window
            p_sum = float(pred[fit].sum())
            scale = float(y[fit].sum()) / p_sum if k and p_sum > 0 else 1.0
            scored = ~np.isin(sensors, chosen) & ~window
            rows.append(
                {
                    "k": k,
                    "rep": rep,
                    "scale": scale,
                    "wape": wape(y[scored], scale * pred[scored]),
                    "mae_log": mae_log(y[scored], scale * pred[scored]),
                }
            )
    return pd.DataFrame(rows)
