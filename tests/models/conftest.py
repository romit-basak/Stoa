"""Shared fixtures for flow model tests: tiny synthetic data, no real parquet files needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def tiny_hourly() -> pd.DataFrame:
    """3 sensors × 100 hours of synthetic data with all columns the model reads."""
    rng = np.random.default_rng(0)
    n = 300  # 3 sensors × 100 hours

    location_ids = np.tile([1, 2, 3], 100)
    base_hour = pd.Timestamp("2023-01-01", tz="UTC")
    # Each sensor gets 100 timestamps spaced 4 hours apart → ~16.7 days per sensor.
    # Sensors are offset by 60 days so they don't overlap. With a 7-day calibration
    # window (~42 timestamps), ~58 rows per sensor fall outside it and are scoreable.
    step_h = 4
    sensor_offset_h = {1: 0, 2: 60 * 24, 3: 120 * 24}
    ts_list = [
        base_hour + pd.Timedelta(hours=sensor_offset_h[location_ids[i]] + (i // 3) * step_h)
        for i in range(n)
    ]
    ts = pd.array(ts_list, dtype="datetime64[us, UTC]")

    df = pd.DataFrame(
        {
            "location_id": location_ids.astype("int32"),
            "ts_utc": ts,
            "ts_local": ts,
            "pedestriancount": rng.integers(10, 500, n).astype("int32"),
            "suspect_outage_zero": np.zeros(n, dtype=bool),
            "has_location": np.ones(n, dtype=bool),
            "location_type": pd.array(["Outdoor"] * n, dtype="str"),
            "needs_review": pd.array([False] * n, dtype="boolean"),
            "post_metro_tunnel": np.zeros(n, dtype=bool),
            # time features
            "hour_local": (np.arange(n) % 24).astype("int8"),
            "weekday": rng.integers(0, 7, n).astype("int8"),
            "is_weekend": rng.integers(0, 2, n).astype(bool),
            "is_public_holiday": np.zeros(n, dtype=bool),
            "school_term": pd.array([True] * n, dtype="boolean"),
            "uni_teaching": pd.array([True] * n, dtype="boolean"),
            "uni_exams": pd.array([False] * n, dtype="boolean"),
            "sun_elevation_deg": rng.uniform(-10, 60, n).astype("float32"),
            "covid_period": np.zeros(n, dtype=bool),
            # weather features
            "wx_temperature_2m": rng.uniform(5, 35, n).astype("float32"),
            "wx_apparent_temperature": rng.uniform(3, 38, n).astype("float32"),
            "wx_relative_humidity_2m": rng.uniform(30, 90, n).astype("float32"),
            "wx_cloud_cover": rng.uniform(0, 100, n).astype("float32"),
            "wx_wind_speed_10m": rng.uniform(0, 20, n).astype("float32"),
            "wx_is_day": rng.uniform(0, 1, n).astype("float32"),
            "wx_precipitation": rng.uniform(0, 5, n).astype("float32"),
            "wx_rain": rng.uniform(0, 5, n).astype("float32"),
            "wx_shortwave_radiation": rng.uniform(0, 800, n).astype("float32"),
            # terrain
            "slope_pct": rng.uniform(0, 10, n).astype("float32"),
            "slope_pct_local_mean": rng.uniform(0, 8, n).astype("float32"),
            "elevation_m": rng.uniform(0, 50, n).astype("float32"),
            # derived (normally added by training_frame.add_derived)
            "shade_portable": rng.uniform(0, 1, n).astype("float32"),
            "shade_rich": rng.uniform(0, 1, n).astype("float32"),
            "event_pre": np.zeros(n, dtype=bool),
            "event_during": np.zeros(n, dtype=bool),
            "event_post": np.zeros(n, dtype=bool),
            "event_dist_m": np.full(n, 3000.0, dtype="float32"),
            "event_day_dist_m": np.full(n, 3000.0, dtype="float32"),
        }
    )
    return df


@pytest.fixture()
def tiny_osm() -> pd.DataFrame:
    """3-row OSM sensor features with a representative subset of real column names."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "location_id": [1, 2, 3],
            "met_betweenness_400": rng.uniform(0, 1, 3).astype("float32"),
            "met_betweenness_1600": rng.uniform(0, 1, 3).astype("float32"),
            "ang_choice_400": rng.uniform(0, 1, 3).astype("float32"),
            "ang_choice_1600": rng.uniform(0, 1, 3).astype("float32"),
            "ang_integration_400": rng.uniform(0, 5, 3).astype("float32"),
            "ang_integration_1600": rng.uniform(0, 5, 3).astype("float32"),
            "junctions_400": rng.integers(0, 20, 3).astype("float32"),
            "poi_food_drink_200": rng.integers(0, 10, 3).astype("float32"),
            "building_coverage": rng.uniform(0, 1, 3).astype("float32"),
            "tram_stop_dist_m": rng.uniform(50, 500, 3).astype("float32"),
            "recent_station_dist_m": rng.uniform(100, 2000, 3).astype("float32"),
            "in_main_component": np.ones(3, dtype=bool),
        }
    ).set_index("location_id")


@pytest.fixture()
def model_cfg() -> dict:
    """Minimal model config for tests (no file I/O)."""
    return {
        "training": {"folds": 2, "seed": 0, "calibration_days": 7},
        "default_params": {
            "max_iter": 10,
            "learning_rate": 0.1,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
            "random_state": 0,
        },
        "gates": {
            "mae_log_improvement_pct": 15.0,
            "interval_coverage_low": 0.75,
            "interval_coverage_high": 0.85,
        },
        "calibration_curve": {"k_values": [1, 2], "reps": 3, "seed": 42},
        "transfer": {"metro_tunnel_date": "2025-11-30"},
    }
