"""Tests for src/models/flow/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.flow.features import calibrate, calibration_window


def test_calibration_window_basic():
    """Each sensor's first N days are marked True."""
    sensors = np.array([1, 1, 1, 2, 2])
    ts = pd.Series(
        pd.to_datetime(
            ["2023-01-01", "2023-01-02", "2023-01-05", "2023-01-01", "2023-01-04"]
        ).tz_localize("UTC")
    )
    window = calibration_window(sensors, ts, days=3)
    # sensor 1: days 0, 1 are in window; day 4 is out
    assert window[0] and window[1] and not window[2]
    # sensor 2: day 0 is in window; day 3 is out
    assert window[3] and not window[4]


def test_calibrate_scales_predictions():
    """calibrate() should scale predictions so they match observed totals in the window."""
    sensors = np.array([1, 1, 1])
    y = np.array([10.0, 20.0, 30.0])
    pred = np.array([5.0, 10.0, 15.0])  # half of actual
    window = np.array([True, True, False])

    out = calibrate(pred, y, sensors, window)
    # In-window sum of y = 30, in-window sum of pred = 15 → scale 2
    assert np.isclose(out[0], 10.0)
    assert np.isclose(out[1], 20.0)
    # Out-of-window row also scaled
    assert np.isclose(out[2], 30.0)


def test_calibrate_zero_pred_no_crash():
    """A sensor with zero predicted pedestrians in the window keeps its predictions."""
    sensors = np.array([1, 1])
    y = np.array([10.0, 10.0])
    pred = np.array([0.0, 0.0])
    window = np.array([True, False])
    out = calibrate(pred, y, sensors, window)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# Smoke test: imports from training_frame are re-exported correctly
# ---------------------------------------------------------------------------

def test_design_matrix_re_export():
    """design_matrix is importable from features and callable on tiny data."""
    from src.models.flow.features import design_matrix

    osm_raw = pd.DataFrame(
        {
            "location_id": [1, 2, 3],
            "met_betweenness_1600": [0.1, 0.2, 0.3],
        }
    )
    osm = osm_raw.set_index("location_id")

    hourly = pd.DataFrame(
        {
            "location_id": pd.array([1, 2, 3], dtype="int32"),
            "post_metro_tunnel": [False, False, False],
        }
    )
    # row_cols = [] so the matrix is just the OSM cols
    idx = np.arange(3)
    x = design_matrix(hourly, idx, osm, station_cap_m=2000.0, row_cols=[])
    assert x.shape == (3, 1)


def test_portable_flag_does_not_crash(tiny_hourly, tiny_osm):
    """sensor_features with portable kwarg doesn't crash (future-proofing hook)."""
    from src.models.flow.features import sensor_features
    # tiny_osm is already indexed; pass a DataFrame with location_id column
    osm_reset = tiny_osm.reset_index()
    # The portable kwarg is forwarded to training_frame.sensor_features
    # This just checks it doesn't raise
    _ = sensor_features(osm_reset, variant="portable")
