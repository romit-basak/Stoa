from datetime import UTC, datetime
from typing import ClassVar

import pandas as pd
import pytest
import yaml

from src.ingest.common import REPO_ROOT
from src.pipeline import validate as v

NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)


@pytest.fixture(scope="module")
def cfg():
    return v.load_validation_config()


def make_table(sensors=(1, 2, 3), days=10) -> pd.DataFrame:
    """A small clean hourly table: live hours up to NOW, every column the checks read."""
    hours = pd.date_range(end=pd.Timestamp(NOW).floor("h"), periods=days * 24, freq="h")
    df = pd.DataFrame(
        [(s, t) for s in sensors for t in hours], columns=["location_id", "ts_utc"]
    )
    df["location_id"] = df["location_id"].astype("int32")
    local = df["ts_utc"].dt.tz_convert("Australia/Melbourne")
    df["ts_local"] = local
    df["date_local"] = local.dt.tz_localize(None).dt.normalize()
    hour = local.dt.hour  # int64: arithmetic on the int8 column would overflow
    weekday = local.dt.weekday
    df["hour_local"] = hour.astype("int8")
    df["source"] = pd.Categorical(["live"] * len(df), categories=["archive", "live"])
    # A daily rhythm that never repeats one value for long and has no outliers.
    df["pedestriancount"] = (100 + 50 * hour + df.index % 7).astype("int32")
    df["direction_1"] = (df["pedestriancount"] // 2).astype("float32")
    df["direction_2"] = (df["pedestriancount"] - df["direction_1"]).astype("float32")
    df["suspect_outage_zero"] = False
    df["weekday"] = weekday.astype("int8")
    df["hour_of_week"] = (weekday * 24 + hour).astype("int16")
    df["is_weekend"] = df["weekday"] >= 5
    df["is_public_holiday"] = False
    df["school_term"] = pd.array([True] * len(df), dtype="boolean")
    df["uni_teaching"] = pd.array([True] * len(df), dtype="boolean")
    df["uni_exams"] = pd.array([False] * len(df), dtype="boolean")
    df["sun_elevation_deg"] = (hour * 5 - 40).astype("float32")
    df["covid_period"] = False
    df["post_metro_tunnel"] = True
    for col, value in {
        "wx_temperature_2m": 18,
        "wx_apparent_temperature": 17,
        "wx_relative_humidity_2m": 60,
        "wx_cloud_cover": 40,
        "wx_wind_speed_10m": 12,
        "wx_is_day": 1,
        "wx_precipitation": 0,
        "wx_rain": 0,
        "wx_shortwave_radiation": 300,
    }.items():
        df[col] = pd.Series(value, index=df.index, dtype="float32")
    df["sensor_description"] = "Sensor " + df["location_id"].astype(str)
    df["location_type"] = "Outdoor"
    df["segment_id"] = df["location_id"].astype(str) + "-0"
    df["match_method"] = "auto"
    df["needs_review"] = pd.array([False] * len(df), dtype="boolean")
    df["slope_pct"] = pd.Series(2.5, index=df.index, dtype="float32")
    df["slope_pct_local_mean"] = pd.Series(2.0, index=df.index, dtype="float32")
    df["elevation_m"] = pd.Series(20.0, index=df.index, dtype="float32")
    df["place_source"] = pd.Categorical(
        ["footpath_steepness"] * len(df), categories=["footpath_steepness", "ga_dem_5m"]
    )
    df["has_location"] = True
    return df


def failures(checks, severity=None):
    return {
        c.name: c.detail
        for c in checks
        if not c.passed and (severity is None or c.severity == severity)
    }


def run(df, cfg, schema=None, previous=None):
    checks, stats = v.validate(
        df, cfg, schema or v.infer_schema(make_table()), previous, NOW
    )
    return checks, stats


def test_clean_table_passes_every_check(cfg):
    checks, _ = run(make_table(), cfg)
    assert failures(checks) == {}


def test_missing_and_retyped_columns_are_schema_errors(cfg):
    df = make_table().drop(columns=["wx_rain"])
    df["hour_local"] = df["hour_local"].astype("int64")
    errors = failures(run(df, cfg)[0], v.ERROR)
    assert "wx_rain" in errors["schema: columns"]
    assert "hour_local is int64, expected int8" in errors["schema: types"]


def test_duplicate_sensor_hours_are_an_error(cfg):
    df = make_table()
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    assert "keys: unique location_id+ts_utc" in failures(run(df, cfg)[0], v.ERROR)


def test_negative_count_is_an_error(cfg):
    df = make_table()
    df.loc[5, "pedestriancount"] = -3
    errors = failures(run(df, cfg)[0], v.ERROR)
    assert "1 rows" in errors["range: pedestriancount in [0, 30000]"]


def test_missing_key_is_an_error(cfg):
    df = make_table()
    df["pedestriancount"] = df["pedestriancount"].astype("float64")
    df.loc[0, "pedestriancount"] = None
    assert "present: pedestriancount" in failures(run(df, cfg)[0], v.ERROR)


def test_value_outside_allowed_set_is_an_error(cfg):
    df = make_table()
    df["source"] = df["source"].cat.add_categories(["test"])
    df.loc[0, "source"] = "test"
    assert "allowed: source in ['archive', 'live']" in failures(
        run(df, cfg)[0], v.ERROR
    )


def test_segment_may_be_missing_only_for_sensors_without_a_location(cfg):
    df = make_table()
    no_location = df["location_id"] == 3
    df.loc[no_location, "has_location"] = False
    for col in (
        "segment_id",
        "location_type",
        "sensor_description",
        "slope_pct",
        "elevation_m",
    ):
        df.loc[no_location, col] = None
    assert failures(run(df, cfg)[0], v.ERROR) == {}

    df.loc[0, "segment_id"] = None  # sensor 1 has a location
    assert "present: segment_id where has_location == True" in failures(run(df, cfg)[0])


def test_a_few_missing_weather_hours_pass_but_many_only_warn(cfg):
    df = make_table()
    df.loc[:3, "wx_temperature_2m"] = None  # 4 of 720 rows: under 1%
    assert failures(run(df, cfg)[0]) == {}

    df.loc[:100, "wx_temperature_2m"] = None
    checks, _ = run(df, cfg)
    assert "present: wx_temperature_2m (>= 99%)" in failures(checks, v.WARNING)
    assert failures(checks, v.ERROR) == {}


STUCK = "anomaly: new runs of the same count (>= 10) for 6+ hours"
OUTLIER = "anomaly: new counts above 4x the sensor's 99.9th percentile"


def test_stuck_sensor_warns_but_small_overnight_repeats_do_not(cfg):
    df = make_table()
    s1 = df.index[df["location_id"] == 1]
    df.loc[s1[-30:-24], "pedestriancount"] = 1  # six quiet hours of 1: normal
    assert failures(run(df, cfg)[0]) == {}

    df.loc[s1[-20:-12], "pedestriancount"] = (
        420  # eight identical busy hours, this week
    )
    assert "sensors [1]" in failures(run(df, cfg)[0], v.WARNING)[STUCK]


def test_stuck_run_before_the_new_hours_is_counted_but_does_not_warn(cfg):
    df = make_table()
    s1 = df.index[df["location_id"] == 1]
    df.loc[s1[10:18], "pedestriancount"] = 420  # nine days ago
    checks, _ = run(df, cfg)  # no previous run: the new hours are the last 7 days
    assert failures(checks) == {}
    assert next(c for c in checks if c.name == STUCK).observed["earlier_runs"] == 1


def test_new_outlier_warns(cfg):
    # 60 days: with only a few hundred hours, the 99.9th percentile sits between the top two
    # values, so a lone outlier raises its own threshold. Real sensors have thousands.
    df = make_table(days=60)
    df.loc[df.index[df["location_id"] == 1][-5], "pedestriancount"] = (
        29000  # ~20x usual
    )
    checks, _ = run(df, cfg)
    assert OUTLIER in failures(checks, v.WARNING)
    assert failures(checks, v.ERROR) == {}


def test_outlier_before_the_last_run_does_not_warn_again(cfg):
    df = make_table(days=60)
    df.loc[df.index[df["location_id"] == 1][-5], "pedestriancount"] = 29000
    # The previous good run already held that hour, so it isn't new.
    previous = v.summary(v.profile(df), NOW)
    checks, _ = run(df, cfg, previous=previous)
    assert OUTLIER not in failures(checks)
    assert next(c for c in checks if c.name == OUTLIER).observed["earlier_rows"] == 1


def test_conditions_are_gx_condition_objects():
    cond = v.condition(["has_location", "==", True])
    assert isinstance(cond, v.Condition)
    assert (cond.column.name, cond.operator, cond.parameter) == (
        "has_location",
        "==",
        True,
    )
    with pytest.raises(ValueError, match="unknown operator"):
        v.condition(["has_location", "is", True])


@pytest.mark.filterwarnings("error:.*row_condition:DeprecationWarning")
@pytest.mark.filterwarnings("error:.*condition_parser:DeprecationWarning")
def test_suite_uses_no_conditions_gx_2_removes(cfg):
    v.run_expectations(make_table(), cfg)


def test_warning_email_lists_only_warnings():
    checks = [
        v.Check("passing check", v.WARNING, True),
        v.Check("anomaly: x", v.WARNING, False, "3 rows"),
        v.Check("schema: columns", v.ERROR, True),
    ]
    subject, body = v.warning_email(checks, "scheduled__2026-10-20")
    assert subject == "[Stoa] validate: 1 data warning"
    assert "<b>anomaly: x</b>: 3 rows" in body
    assert "scheduled__2026-10-20" in body
    assert "passing check" not in body
    assert v.warning_email(checks[:1]) is None


class FakeHook:
    sent: ClassVar[list[dict]] = []

    def __init__(self, smtp_conn_id):
        assert smtp_conn_id == "smtp_default"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def send_email_smtp(self, **kw):
        FakeHook.sent.append(kw)


@pytest.fixture
def smtp(monkeypatch):
    """Replace Airflow's SmtpHook (not installed in .venv) with FakeHook."""
    import sys
    import types

    FakeHook.sent = []
    module = types.ModuleType("airflow.providers.smtp.hooks.smtp")
    module.SmtpHook = FakeHook
    monkeypatch.setitem(sys.modules, "airflow.providers.smtp.hooks.smtp", module)
    monkeypatch.setenv("STOA_ALERT_EMAILS", "a@x.edu, b@x.edu")
    monkeypatch.setenv("STOA_SMTP_USER", "stoa@gmail.com")
    return FakeHook


WARNED = [v.Check("anomaly: x", v.WARNING, False, "3 rows")]


def test_warnings_are_emailed_to_the_team(smtp):
    assert v.send_warning_alert(WARNED) is True
    (sent,) = smtp.sent
    assert sent["to"] == ["a@x.edu", "b@x.edu"]
    assert sent["from_email"] == "stoa@gmail.com"
    assert sent["subject"] == "[Stoa] validate: 1 data warning"


def test_no_email_without_warnings_or_recipients(smtp, monkeypatch):
    assert v.send_warning_alert([v.Check("ok", v.WARNING, True)]) is False
    monkeypatch.setenv("STOA_ALERT_EMAILS", "")
    assert v.send_warning_alert(WARNED) is False
    assert smtp.sent == []


def test_failed_send_is_logged_not_raised(smtp, monkeypatch, caplog):
    def broken(self, **kw):
        raise OSError("smtp down")

    monkeypatch.setattr(FakeHook, "send_email_smtp", broken)
    assert v.send_warning_alert(WARNED) is False
    assert "could not email the warnings" in caplog.text


def test_missing_night_hours_are_normal_but_daytime_gaps_warn(cfg):
    df = make_table()
    night = df["hour_local"].isin([1, 2, 3, 4]) & (df["location_id"] == 2)
    assert failures(run(df[~night], cfg)[0]) == {}

    recent_day = (
        (df["location_id"] == 2)
        & df["hour_local"].between(8, 19)
        & (df["ts_utc"] > pd.Timestamp(NOW) - pd.Timedelta(days=2))
    )
    warnings = failures(run(df[~recent_day], cfg)[0], v.WARNING)
    assert any(name.startswith("anomaly: each sensor reported") for name in warnings)


def test_stale_data_warns(cfg):
    df = make_table()
    df = df[df["ts_utc"] < pd.Timestamp(NOW) - pd.Timedelta(days=6)]
    assert "anomaly: newest hour at most 5 days old" in failures(
        run(df, cfg)[0], v.WARNING
    )


def test_lost_rows_since_the_last_run_are_an_error(cfg):
    df = make_table()
    _, stats = run(df, cfg)
    previous = v.summary(stats, NOW)

    assert failures(run(df, cfg, previous=previous)[0]) == {}
    fewer = df.iloc[: len(df) // 2]
    assert "drift: rows dropped by at most 1%" in failures(
        run(fewer, cfg, previous=previous)[0], v.ERROR
    )


def test_more_missing_values_and_a_shifted_mean_warn(cfg):
    df = make_table()
    previous = v.summary(run(df, cfg)[1], NOW)
    df["wx_cloud_cover"] = df["wx_cloud_cover"].where(df.index % 10 != 0)  # 10% missing
    df["pedestriancount"] = df["pedestriancount"] * 2
    warnings = failures(run(df, cfg, previous=previous)[0], v.WARNING)
    assert (
        "wx_cloud_cover"
        in warnings["drift: missing values up by at most 5 points per column"]
    )
    assert "drift: mean hourly count changed by at most 20%" in warnings


def test_profile_covers_every_column():
    df = make_table()
    stats = v.profile(df)
    assert stats["rows"] == len(df)
    assert stats["sensors"] == 3
    assert set(stats["columns"]) == set(df.columns)
    assert stats["columns"]["pedestriancount"]["min"] == df["pedestriancount"].min()
    assert stats["columns"]["is_weekend"]["true_share"] == pytest.approx(
        df["is_weekend"].mean()
    )


def test_reports_are_written_and_only_good_runs_join_the_history(cfg, tmp_path):
    good, stats = run(make_table(), cfg)
    v.write_reports(tmp_path, good, stats, NOW)
    bad = [*good, v.Check("made up", v.ERROR, False, "broken")]
    v.write_reports(tmp_path, bad, stats, NOW)

    history = tmp_path / "counts_hourly_stats_history.jsonl"
    assert len(history.read_text().splitlines()) == 1
    assert v.last_summary(history)["rows"] == len(make_table())
    report = yaml.safe_load((tmp_path / "counts_hourly_validation.json").read_text())
    assert report["passed"] is False
    assert report["errors"] == 1
    assert (tmp_path / "counts_hourly_stats.json").is_file()


def test_schema_round_trips(tmp_path):
    df = make_table()
    path = tmp_path / "schema.yaml"
    v.write_schema(df, path)
    assert yaml.safe_load(path.read_text()) == v.infer_schema(df)


def test_config_only_names_columns_in_the_committed_schema(cfg):
    schema = yaml.safe_load((REPO_ROOT / cfg["schema"]).read_text())["columns"]
    named = {
        *cfg["keys"],
        *cfg["never_missing"],
        *cfg["required_when"],
        *cfg["ranges"],
        *cfg["allowed_values"],
    }
    assert named <= set(schema)
