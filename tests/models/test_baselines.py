"""Tests for src/models/flow/baselines.py."""

from __future__ import annotations

import numpy as np

from src.features.training_frame import fold_plan
from src.models.flow.baselines import betweenness_baseline, space_syntax_baseline


def test_betweenness_baseline_shape(tiny_hourly, tiny_osm):
    sensors = tiny_hourly["location_id"].to_numpy()
    plan = fold_plan(sensors, n_folds=2, train_rows=600, seed=0)
    oof, _ = betweenness_baseline(
        tiny_hourly, tiny_osm, sensors, plan, calibration_days=7
    )
    assert oof.shape == (len(tiny_hourly),)


def test_betweenness_nonnegative(tiny_hourly, tiny_osm):
    sensors = tiny_hourly["location_id"].to_numpy()
    plan = fold_plan(sensors, n_folds=2, train_rows=600, seed=0)
    oof, _ = betweenness_baseline(tiny_hourly, tiny_osm, sensors, plan, calibration_days=7)
    # Rows not in any test fold may be NaN; tested rows must be >= 0
    tested = ~np.isnan(oof)
    assert np.all(oof[tested] >= 0.0)


def test_space_syntax_baseline_shape(tiny_hourly, tiny_osm):
    sensors = tiny_hourly["location_id"].to_numpy()
    plan = fold_plan(sensors, n_folds=2, train_rows=600, seed=0)
    oof, _ = space_syntax_baseline(
        tiny_hourly, tiny_osm, sensors, plan, calibration_days=7
    )
    assert oof.shape == (len(tiny_hourly),)


def test_space_syntax_nonnegative(tiny_hourly, tiny_osm):
    sensors = tiny_hourly["location_id"].to_numpy()
    plan = fold_plan(sensors, n_folds=2, train_rows=600, seed=0)
    oof, _ = space_syntax_baseline(tiny_hourly, tiny_osm, sensors, plan, calibration_days=7)
    tested = ~np.isnan(oof)
    assert np.all(oof[tested] >= 0.0)
