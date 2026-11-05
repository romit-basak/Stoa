"""Baseline ladder rungs for comparing against the learned flow model.

Two baselines, both run with the same GroupKFold plan and calibration window
as the main model so errors are measured on identical held-out rows.

Rung 1 — betweenness: shortest-path betweenness centrality (met_betweenness_1600)
  scaled to each sensor's mean count. No learning; just the network's geometry.

Rung 2 — Space Syntax: Ridge regression on all ang_* and met_* network columns
  (at three radii) plus hour_local and weekday, with per-sensor scale calibration.
  This is the benchmark the learned model must beat by >=15% on MAE-log counts
  (docs/PROJECT_CONTEXT.md §13).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.flow.features import calibrate, calibration_window, fold_plan


def _space_syntax_cols(osm: pd.DataFrame) -> list[str]:
    """All network-metric columns in sensor_osm_features, at every radius."""
    return [c for c in osm.columns if c.startswith(("ang_", "met_", "junctions_"))]


def betweenness_baseline(
    hourly: pd.DataFrame,
    osm: pd.DataFrame,
    fold_groups: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
    calibration_days: float = 28,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """OOF predictions from met_betweenness_1600 scaled to each training fold's mean count.

    For every test fold: predict = sensor_mean_count * (betweenness / training_mean_betweenness),
    then apply the same per-sensor scale calibration used by the main model.

    Returns (oof_preds, [test_index_array per fold]).
    """
    sensors = hourly["location_id"].to_numpy()
    y = hourly["pedestriancount"].to_numpy(dtype=float)
    ts = hourly["ts_utc"]
    window = calibration_window(sensors, ts, calibration_days)

    # betweenness per sensor from the OSM features (sensor-indexed)
    bw_col = "met_betweenness_1600"
    if bw_col not in osm.columns:
        # fall back to whatever betweenness is available
        bw_cols = [c for c in osm.columns if c.startswith("met_betweenness")]
        bw_col = bw_cols[-1] if bw_cols else None

    bw_per_sensor = (
        osm[bw_col].fillna(0.0).to_numpy(dtype=float) if bw_col else np.ones(len(osm))
    )
    bw_map = dict(zip(osm.index.tolist(), bw_per_sensor))

    oof = np.full(len(hourly), np.nan)
    test_indices: list[np.ndarray] = []

    for train_pos, test_pos in plan:
        train_sensors = sensors[train_pos]
        train_y = y[train_pos]
        train_bw = np.array([bw_map.get(s, 0.0) for s in train_sensors])

        mean_bw_train = train_bw.mean() if train_bw.mean() > 0 else 1.0
        mean_count_train = train_y.mean() if len(train_y) else 1.0
        ratio = mean_count_train / mean_bw_train

        test_sensors = sensors[test_pos]
        test_bw = np.array([bw_map.get(s, 0.0) for s in test_sensors])
        pred = test_bw * ratio

        # per-sensor calibration using the first N days
        pred_cal = calibrate(pred, y[test_pos], test_sensors, window[test_pos])
        oof[test_pos] = pred_cal
        test_indices.append(test_pos)

    return np.maximum(oof, 0.0), test_indices


def space_syntax_baseline(
    hourly: pd.DataFrame,
    osm: pd.DataFrame,
    fold_groups: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
    calibration_days: float = 28,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """OOF predictions from Ridge on all ang_*/met_*/junctions_* columns + hour + weekday.

    Trained on log1p(y) to keep the problem linear; predictions are expm1-ed back.
    Per-sensor scale calibration applied after.

    Returns (oof_preds, [test_index_array per fold]).
    """
    sensors = hourly["location_id"].to_numpy()
    y = hourly["pedestriancount"].to_numpy(dtype=float)
    ts = hourly["ts_utc"]
    window = calibration_window(sensors, ts, calibration_days)

    ss_cols = _space_syntax_cols(osm)
    # Row-level time features added as sensor-constant in this simplified baseline
    time_cols = ["hour_local", "weekday"]
    available_time = [c for c in time_cols if c in hourly.columns]

    oof = np.full(len(hourly), np.nan)
    test_indices: list[np.ndarray] = []

    for train_pos, test_pos in plan:
        test_sensors = sensors[test_pos]

        # Build feature matrices: [osm network metrics per row] + [time features]
        def _build_x(positions: np.ndarray) -> np.ndarray:
            part_sensors = sensors[positions]
            seg = osm[ss_cols].reindex(part_sensors).to_numpy(dtype="float32")
            parts = [seg]
            if available_time:
                time_part = hourly[available_time].iloc[positions].to_numpy(dtype="float32")
                parts.append(time_part)
            return np.hstack(parts)

        x_train = _build_x(train_pos)
        y_train = np.log1p(y[train_pos])

        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        # NaN in network metrics (sensors without coverage) → 0 after scaling
        x_train_sc = np.nan_to_num(x_train_sc, nan=0.0)

        model = Ridge(alpha=1.0)
        model.fit(x_train_sc, y_train)

        x_test_sc = np.nan_to_num(scaler.transform(_build_x(test_pos)), nan=0.0)
        pred_log = model.predict(x_test_sc)
        pred = np.expm1(np.maximum(pred_log, 0.0))

        pred_cal = calibrate(pred, y[test_pos], test_sensors, window[test_pos])
        oof[test_pos] = pred_cal
        test_indices.append(test_pos)

    return np.maximum(oof, 0.0), test_indices


def run_baselines(
    hourly: pd.DataFrame,
    osm: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Run both baseline rungs and return their OOF predictions.

    Expects hourly to already be filtered to eligible rows (no outages, located outdoors).
    """
    tcfg = cfg["training"]
    sensors = hourly["location_id"].to_numpy()
    plan = fold_plan(sensors, tcfg["folds"], 600_000, tcfg["seed"])
    cal_days = tcfg["calibration_days"]

    bw_pred, _ = betweenness_baseline(hourly, osm, sensors, plan, cal_days)
    ss_pred, _ = space_syntax_baseline(hourly, osm, sensors, plan, cal_days)
    return {"betweenness": bw_pred, "space_syntax": ss_pred}
