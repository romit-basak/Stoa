"""The API's endpoints with small fake tables, a fake model and fake weather (no data or
network needed, so these run in CI). Feature correctness is test_parity.py's job."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import model as model_mod
from src.api.app import create_app
from src.api.features import STEEPNESS, SiteTables
from src.api.weather import ForecastWeather, to_frame
from src.features import counts_hourly as ch
from src.features import training_frame as tf
from src.features.common import load_features_config
from src.ingest.common import REPO_ROOT, load_config

SEGMENTS = ["100-0", "100-1", "200-0"]
WEATHER_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "is_day",
]


@pytest.fixture
def tables(tmp_path):
    data_cfg, fcfg = load_config(), load_features_config()
    cols = tf.segment_features("rich")
    rng = np.random.default_rng(0)
    segments = pd.DataFrame(
        rng.random((len(SEGMENTS), len(cols))).astype("float32"),
        index=pd.Index(SEGMENTS, name="segment_id"),
        columns=cols,
    )
    steep = pd.DataFrame(
        {"slope_pct": 1.0, "slope_pct_local_mean": 1.5, "elevation_m": 20.0},
        index=segments.index,
    ).astype("float32")
    steep["place_source"] = ["sensor", "footpath_steepness", "ga_dem_5m"]
    cal = REPO_ROOT / "configs" / "calendars"
    school = ch.school_term_ranges(cal / "vic_school_terms.yaml")
    uni = ch.uni_ranges(cal / "unimelb_teaching_periods.yaml")
    return SiteTables(
        site="melbourne",
        data_cfg=data_cfg,
        fcfg=fcfg,
        feats=tmp_path,  # no shade or venue tables here: their defaults apply
        segments=segments,
        steepness=steep,
        shade=None,
        holidays=pd.to_datetime(pd.Series(["2026-12-25"])),
        school=school,
        uni=uni,
        center=(144.9631, -37.8136),
        timezone="Australia/Melbourne",
        station_cap_m=3000.0,
        calendar_years=school[1] & uni[1],
    )


INTERVAL_VARS = ["precipitation", "rain", "shortwave_radiation"]


def fake_weather(
    data_cfg, start="2026-11-20", days=10, null_tail=0, drop=None, fail=False
):
    """A forecast of `days` days from `start` (UTC). Like Open-Meteo's, the last `null_tail`
    hours can come back empty; `drop` leaves a variable out; `fail` makes the fetch fail."""

    def fetch(params):
        if fail:
            raise ConnectionError("open-meteo unreachable")
        hours = pd.date_range(start, periods=24 * days, freq="h")
        hourly = {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist()}
        for v in WEATHER_VARS:
            if v == drop:
                continue
            vals = list(np.linspace(0, 1, len(hours)))
            if null_tail and v != "is_day":
                vals[-null_tail:] = [None] * null_tail
            hourly[v] = vals
        return {"utc_offset_seconds": 0, "hourly": hourly}

    return ForecastWeather(
        data_cfg, "melbourne", fetch=fetch, interval_vars=INTERVAL_VARS
    )


class HourModel:
    """predicted = 10 x the local hour, so the test can check rows line up."""

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        p = 10.0 * x["hour_local"].to_numpy()
        return pd.DataFrame({"predicted": p, "lower": p * 0.5, "upper": p * 2})


def loaded(variant="rich"):
    return model_mod.LoadedModel(
        variant,
        "models:/stoa-flow-rich@champion",
        model_mod.expected_columns(variant),
        HourModel(),
        "stoa-flow-rich",
        "3",
        {"log_mae_cv": 0.87},
    )


@pytest.fixture
def client(tables):
    app = create_app(tables, {"rich": loaded()}, fake_weather(tables.data_cfg))
    with TestClient(app) as c:
        yield c


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "variants": ["rich"]}
    info = client.get("/model").json()["models"]["rich"]
    assert info["version"] == "3" and info["features"] == 163


def test_health_is_503_without_a_model(tables):
    app = create_app(
        tables, {"rich": "OSError: no such model"}, fake_weather(tables.data_cfg)
    )
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.get("/model").json()["models"]["rich"] == {
            "error": "OSError: no such model"
        }


def test_predict_returns_every_segment_hour(client):
    hours = ["2026-11-24T08:00", "2026-11-24T09:00", "2026-11-24T17:00"]
    r = client.post("/predict", json={"segment_ids": SEGMENTS[:2], "hours": hours})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] == "stoa-flow-rich v3"
    preds = body["predictions"]
    assert [(p["segment_id"], p["hour"]) for p in preds] == [
        (s, h[:16]) for s in SEGMENTS[:2] for h in hours
    ]
    assert [p["predicted_count"] for p in preds[:3]] == [80.0, 90.0, 170.0]
    assert all(p["lower"] <= p["predicted_count"] <= p["upper"] for p in preds)
    src = body["sources"]
    assert src["steepness"] == {"100-0": "sensor", "100-1": "footpath_steepness"}
    assert src["weather"].startswith("Open-Meteo forecast")
    assert "not a measured count" in body["note"]


def test_unknown_segment_is_404(client):
    r = client.post(
        "/predict", json={"segment_ids": ["999-9"], "hours": ["2026-11-24T08:00"]}
    )
    assert r.status_code == 404
    assert r.json()["detail"] == {"unknown_segment_ids": ["999-9"]}


def test_hour_outside_the_forecast_is_422(client):
    r = client.post(
        "/predict", json={"segment_ids": SEGMENTS[:1], "hours": ["2027-03-01T08:00"]}
    )
    assert r.status_code == 422 and "forecast" in r.json()["detail"]


def test_spring_forward_hour_is_422(tables):
    # Clocks in Melbourne go from 02:00 to 03:00 on 4 October 2026.
    app = create_app(
        tables, {"rich": loaded()}, fake_weather(tables.data_cfg, start="2026-10-01")
    )
    with TestClient(app) as c:
        r = c.post(
            "/predict",
            json={"segment_ids": SEGMENTS[:1], "hours": ["2026-10-04T02:00"]},
        )
    assert r.status_code == 422 and "clock" in r.json()["detail"]


def test_variant_not_loaded_is_503(client):
    r = client.post(
        "/predict",
        json={
            "variant": "portable",
            "segment_ids": SEGMENTS[:1],
            "hours": ["2026-11-24T08:00"],
        },
    )
    assert r.status_code == 503


@pytest.mark.parametrize(
    "body",
    [
        {"segment_ids": [], "hours": ["2026-11-24T08:00"]},
        {"segment_ids": ["100-0"] * 501, "hours": ["2026-11-24T08:00"]},
        {"segment_ids": ["100-0"] * 200, "hours": ["2026-11-24T08:00"] * 101},
        {"site": "nyc", "segment_ids": ["100-0"], "hours": ["2026-11-24T08:00"]},
    ],
)
def test_request_limits(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_forecast_frame_undoes_the_offset():
    payload = {
        "utc_offset_seconds": 36000,
        "hourly": {
            "time": ["2026-11-24T10:00", "2026-11-24T11:00"],
            "rain": [0.0, 1.0],
        },
    }
    table = to_frame(payload)
    assert table.index[0] == pd.Timestamp("2026-11-24T00:00", tz="UTC")


def test_steepness_columns_are_the_rich_row_features():
    assert set(STEEPNESS) <= set(tf.row_features("rich"))
    assert not set(STEEPNESS) & set(tf.row_features("portable"))


def post(tables, weather, body):
    app = create_app(tables, {"rich": loaded()}, weather)
    with TestClient(app) as c:
        return c.post("/predict", json=body)


def test_hours_with_an_empty_forecast_are_422(tables):
    # The forecast runs to 2026-11-29 23:00 UTC; its last 5 hours are null, as Open-Meteo's
    # are. 2026-11-30 08:00 Melbourne is 2026-11-29 21:00 UTC: inside the range, no values.
    w = fake_weather(tables.data_cfg, null_tail=5)
    r = post(tables, w, {"segment_ids": ["100-0"], "hours": ["2026-11-30T08:00"]})
    assert r.status_code == 422 and "no weather forecast" in r.json()["detail"]
    # 04:00 Melbourne is 17:00 UTC, the last hour whose next-hour row (where rain and
    # radiation are stamped) still has values.
    ok = post(tables, w, {"segment_ids": ["100-0"], "hours": ["2026-11-30T04:00"]})
    assert ok.status_code == 200, ok.text
    late = post(tables, w, {"segment_ids": ["100-0"], "hours": ["2026-11-30T05:00"]})
    assert late.status_code == 422


@pytest.mark.parametrize(
    "hours, words",
    [
        (["2026-11-24T08:00+11:00", "2026-11-24T09:00+10:00"], "UTC offset"),
        (["2026-11-24T08:00+11:00"], "UTC offset"),
        (["2026-11-24T08:45"], "whole hours"),
    ],
)
def test_offsets_and_part_hours_are_422(tables, hours, words):
    r = post(
        tables,
        fake_weather(tables.data_cfg),
        {"segment_ids": ["100-0"], "hours": hours},
    )
    assert r.status_code == 422 and words in r.json()["detail"]


@pytest.mark.parametrize("kwargs", [{"fail": True}, {"drop": "cloud_cover"}])
def test_weather_unavailable_is_503(tables, kwargs):
    w = fake_weather(tables.data_cfg, **kwargs)
    r = post(tables, w, {"segment_ids": ["100-0"], "hours": ["2026-11-24T08:00"]})
    assert r.status_code == 503 and "weather forecast" in r.json()["detail"]


def test_an_alias_is_resolved_to_its_version_before_loading(monkeypatch):
    from types import SimpleNamespace

    import mlflow

    run = SimpleNamespace(
        data=SimpleNamespace(metrics={"log_mae_cv": 0.86, "other": 1.0})
    )

    class Client:
        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == ("stoa-flow-rich", "champion")
            return SimpleNamespace(version=7, run_id="r1")

        def get_run(self, run_id):
            return run

    monkeypatch.setattr(mlflow, "MlflowClient", Client)
    uri, name, version, metrics = model_mod.resolve("models:/stoa-flow-rich@champion")
    assert (uri, name, version) == ("models:/stoa-flow-rich/7", "stoa-flow-rich", "7")
    assert metrics == {"log_mae_cv": 0.86}
    assert model_mod.resolve("/models/rich")[:3] == ("/models/rich", None, None)
