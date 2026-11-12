"""Synthetic flow-model inputs: 14 sensors on three folds plus a holdout precinct."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.features.training_frame import Frame
from src.models.flow.data import Data, flow_config

ROW_COLS = ["hour_local", "weekday", "wx_temperature_2m"]
# 12 fold sensors (4 per fold, each its own spatial group) and two in "carlton".
FOLDS = {s: (s - 1) % 3 for s in range(1, 13)} | {13: -1, 14: -1}
HOLDOUT = {13: "carlton", 14: "carlton"}


def make_data(variant: str = "rich", days: int = 12, seed: int = 0) -> Data:
    rng = np.random.default_rng(seed)
    sensors = np.array(list(FOLDS))
    hours = pd.date_range("2025-11-20", periods=days * 24, freq="h", tz="UTC")
    loc = np.repeat(sensors, len(hours))
    ts = pd.Series(np.tile(hours, len(sensors)))
    local = ts.dt.tz_convert("Australia/Melbourne")
    hour = local.dt.hour.to_numpy()
    weekday = local.dt.day_of_week.to_numpy()
    betweenness = {s: 1000.0 * s for s in sensors}
    base = np.array([5 + betweenness[s] / 400 for s in loc])
    profile = 1 + 3 * np.exp(-((hour - 13) ** 2) / 18)
    y = rng.poisson(base * profile).astype("int32")
    hourly = pd.DataFrame(
        {
            "location_id": loc.astype("int32"),
            "ts_utc": ts.to_numpy(),
            "date_local": local.dt.tz_localize(None).dt.normalize().to_numpy(),
            "pedestriancount": y,
            "post_metro_tunnel": (
                ts >= pd.Timestamp("2025-11-30", tz="UTC")
            ).to_numpy(),
            "hour_local": hour.astype("int8"),
            "weekday": weekday.astype("int8"),
            "hour_of_week": (weekday * 24 + hour).astype("int16"),
            "wx_temperature_2m": rng.normal(18, 4, len(loc)).astype("float32"),
        }
    )
    osm = pd.DataFrame(
        {
            "met_betweenness_1600": [betweenness[s] for s in sensors],
            "ang_choice_800": rng.random(len(sensors)) * 100,
            "met_reach_400": rng.random(len(sensors)) * 50,
            "junctions_400": rng.integers(2, 20, len(sensors)).astype(float),
        },
        index=pd.Index(sensors, name="location_id"),
    ).astype("float32")
    frame = Frame(hourly, osm, ROW_COLS, 2000.0)
    fold = np.array([FOLDS[s] for s in loc])
    holdout = np.array([HOLDOUT.get(s, "") for s in loc], dtype=object)
    return Data(frame, fold, loc.copy(), holdout, "melbourne", variant)


@pytest.fixture()
def data() -> Data:
    return make_data()


@pytest.fixture()
def fcfg() -> dict:
    """The real flow settings, shrunk so a whole run takes seconds."""
    cfg = copy.deepcopy(flow_config())
    cfg["train_rows_per_fold"] = 100_000
    cfg["final_train_rows"] = 100_000
    cfg["calibration_days"] = 3
    cfg["calibration_curve"] = {"k": [1, 3], "reps": 3}
    cfg["tune"].update(n_trials=2, train_rows=2000)
    small = {
        "hgb": {"min_samples_leaf": 5},
        "lgbm": {"min_child_samples": 5},
        "xgb": {"min_child_weight": 1.0},
    }
    for family, extra in small.items():
        cfg["candidates"][family]["params"].update(extra)
    return cfg


@pytest.fixture()
def mlflow_store(tmp_path, monkeypatch):
    uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    # MLflow 3 keeps the file store only behind this switch; it's fine for a throwaway store.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri
