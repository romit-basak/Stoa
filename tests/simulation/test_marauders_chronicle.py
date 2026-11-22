"""Tests for the full-bleed STOA simulation renderer helpers."""

from __future__ import annotations

import numpy as np

from src.simulation.marauders_chronicle.render import (
    VIEW,
    day_phase,
    deduplicate_sensors,
    flow_vein_colors,
    footprint_polygons,
)


def test_day_phase_labels() -> None:
    assert day_phase(2.0) == "Late Night"
    assert day_phase(8.5) == "Morning Commute"
    assert day_phase(13.0) == "Midday Rush"
    assert day_phase(17.5) == "Evening Peak"
    assert day_phase(20.0) == "Evening Dining"


def test_flow_vein_colors_range() -> None:
    rgb = flow_vein_colors(np.array([0.0, 0.55, 1.0]))
    assert rgb.shape == (3, 3)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)


def test_deduplicate_sensors_prefers_observed_and_filters_view() -> None:
    cx = (VIEW[0] + VIEW[2]) / 2
    cy = (VIEW[1] + VIEW[3]) / 2
    sensors = [
        {
            "location_id": 1,
            "name": "No Obs",
            "x": cx,
            "y": cy,
            "observed": [None] * 24,
            "predicted": [500.0] * 24,
        },
        {
            "location_id": 2,
            "name": "Has Obs",
            "x": cx + 5.0,
            "y": cy + 5.0,
            "observed": [100.0] * 24,
            "predicted": [400.0] * 24,
        },
        {
            "location_id": 3,
            "name": "Distinct",
            "x": cx + 200.0,
            "y": cy,
            "observed": [None] * 24,
            "predicted": [300.0] * 24,
        },
        {
            "location_id": 4,
            "name": "Outside",
            "x": VIEW[0] - 500.0,
            "y": cy,
            "observed": [100.0] * 24,
            "predicted": [999.0] * 24,
        },
    ]
    kept = deduplicate_sensors(sensors, view=VIEW, min_dist_m=28.0)
    ids = [s["location_id"] for s in kept]
    assert ids == [2, 3]


def test_footprint_polygons_shape_and_rotation() -> None:
    sole, heel = footprint_polygons(
        np.array([10.0, 20.0]),
        np.array([30.0, 40.0]),
        np.array([0.0, np.pi / 2]),
        scale=10.0,
    )
    assert sole.shape == (2, 9, 2)
    assert heel.shape == (2, 7, 2)
    assert sole[0, 0, 0] > 10.0
    assert sole[1, 0, 1] > 40.0
