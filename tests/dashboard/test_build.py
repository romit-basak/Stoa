"""Tests for the universal multi-city, multi-scenario STOA dashboard builder."""

from __future__ import annotations

import json
from pathlib import Path

from src.dashboard.build import (
    build_dashboard,
    classify_sensor_status,
    compute_scenario_payload,
    make_projector,
    utm_to_latlon,
)


def _melbourne_sensors() -> list[dict]:
    return [
        {
            "location_id": 1,
            "name": "I-Hub Bourke Street Mall - New footpath",
            "x": 320885.0,
            "y": 5812908.0,
            "predicted": [100.0 + h * 10.0 for h in range(24)],
            "observed": [95.0 + h * 10.0 for h in range(24)],
        },
        {
            "location_id": 2,
            "name": "Flinders Street",
            "x": 321020.0,
            "y": 5812445.0,
            "predicted": [200.0 for _ in range(24)],
            "observed": [None for _ in range(24)],
        },
    ]


def _boston_forecast_only_sensors() -> list[dict]:
    return [
        {
            "location_id": 101,
            "name": "Centennial Common",
            "lat": 42.3374,
            "lon": -71.0894,
            "predicted": [80.0 + h * 12.0 for h in range(24)],
            "observed": [None for _ in range(24)],
        },
        {
            "location_id": 102,
            "name": "Curry Student Center",
            "lat": 42.3391,
            "lon": -71.0875,
            "predicted": [140.0 + h * 15.0 for h in range(24)],
            "observed": [None for _ in range(24)],
        },
    ]


def test_utm_to_latlon_north_and_south() -> None:
    # Melbourne (Zone 55S)
    lat_m, lon_m = utm_to_latlon(321020.0, 5812445.0, zone=55, northern=False)
    assert abs(lat_m - (-37.8180)) < 0.003
    assert abs(lon_m - 144.9669) < 0.003
    # NYC (Zone 18N — Midtown Manhattan ~40.758, -73.985)
    lat_n, lon_n = utm_to_latlon(585650.0, 4512400.0, zone=18, northern=True)
    assert 40.70 < lat_n < 40.82
    assert -74.05 < lon_n < -73.90


def test_make_projector_handles_crs_variants() -> None:
    p_melb = make_projector(28355)
    lat, lon = p_melb(321020.0, 5812445.0)
    assert -37.85 < lat < -37.78 and 144.93 < lon < 145.00

    p_wgs = make_projector("EPSG:4326")
    assert p_wgs(-71.0894, 42.3374) == (42.3374, -71.0894)


def test_classify_sensor_status_including_forecast_only() -> None:
    assert classify_sensor_status(0, 0.18, True) == "Healthy"
    assert classify_sensor_status(0, 0.45, True) == "High Drift"
    assert classify_sensor_status(6, 0.15, True) == "Partial Outage"
    assert classify_sensor_status(24, None, True) == "Offline"
    assert classify_sensor_status(24, None, False) == "Forecast Only"


def test_multi_city_dashboard_discovery_and_build(tmp_path: Path) -> None:
    sim_root = tmp_path / "simulation"
    melb = sim_root / "melbourne_cbd"
    boston = sim_root / "neu_boston_event"
    melb.mkdir(parents=True)
    boston.mkdir(parents=True)

    (melb / "meta.json").write_text(
        json.dumps(
            {
                "date": "2026-11-06",
                "model": "stoa-flow-rich v1",
                "run_id": "run-melb",
                "crs": 28355,
                "segments": 30190,
                "sensors": 2,
                "sensor_hours_compared": 24,
                "sensor_wape": 0.24,
                "sensor_log_mae": 0.38,
            }
        ),
        encoding="utf-8",
    )
    (melb / "sensors.json").write_text(
        json.dumps(_melbourne_sensors()), encoding="utf-8"
    )

    (boston / "meta.json").write_text(
        json.dumps(
            {
                "date": "2026-11-15",
                "site_label": "NEU Boston — Event Scenario",
                "model": "stoa-flow-portable v1",
                "run_id": "run-bos",
                "crs": "EPSG:4326",
                "segments": 4800,
                "sensors": 2,
            }
        ),
        encoding="utf-8",
    )
    (boston / "sensors.json").write_text(
        json.dumps(_boston_forecast_only_sensors()), encoding="utf-8"
    )
    (boston / "landmarks.json").write_text(
        json.dumps(
            [
                {
                    "name": "Krentzman Quad",
                    "category": "park",
                    "lat": 42.3398,
                    "lon": -71.0892,
                }
            ]
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "dashboard"
    outputs = build_dashboard(
        sim_root=sim_root,
        video_path=tmp_path / "missing.mp4",
        counts_meta_path=tmp_path / "missing_counts.json",
        bias_csv_path=tmp_path / "missing_bias.csv",
        out_dir=out_dir,
    )
    bundle = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert bundle["project"] == "STOA"
    assert len(bundle["scenarios"]) == 2
    by_id = {s["id"]: s for s in bundle["scenarios"]}

    assert by_id["melbourne_cbd"]["site_label"] == "Melbourne CBD"
    assert by_id["melbourne_cbd"]["has_observed"] is True

    assert by_id["neu_boston_event"]["site_label"] == "NEU Boston — Event Scenario"
    assert by_id["neu_boston_event"]["has_observed"] is False
    assert by_id["neu_boston_event"]["status_counts"]["Forecast Only"] == 2
    assert len(by_id["neu_boston_event"]["landmarks"]) == 1
    assert 42.3 < by_id["neu_boston_event"]["map_center"][0] < 42.4

    html = outputs["html"].read_text(encoding="utf-8")
    assert "scenarioSel" in html
    assert "fitBounds" in html


def test_compute_scenario_payload_dynamic_bounds() -> None:
    payload = compute_scenario_payload(
        scenario_id="neu_boston",
        meta={"date": "2026-11-15", "crs": "EPSG:4326", "segments": 500},
        raw_sensors=_boston_forecast_only_sensors(),
    )
    assert payload["site_label"] == "NEU Boston"
    assert payload["map_bounds"][0][0] <= payload["map_bounds"][1][0]
    assert payload["map_bounds"][0][1] <= payload["map_bounds"][1][1]
