import json

import numpy as np
import pandas as pd
import pytest

from src.features.counts_hourly import (
    covered,
    drop_unplaceable,
    fill_gap,
    flag_outages,
    localize,
    read_wayback,
    read_weather,
    solar_elevation,
    weather_for_counts,
)

TZ = "Australia/Melbourne"
MELB = (144.9631, -37.8136)  # lon, lat


def counts_at(*stamps, sensor=1, count=10):
    return pd.DataFrame(
        {
            "location_id": sensor,
            "ts_local_naive": pd.to_datetime(list(stamps)),
            "pedestriancount": count,
        }
    )


def test_spring_forward_hour_is_dropped():
    # Clocks went from 02:00 to 03:00 on 5 October 2025, so 02:00 never happened.
    c, dropped = localize(
        counts_at("2025-10-05 01:00", "2025-10-05 02:00", "2025-10-05 03:00"), TZ
    )
    assert dropped == 1
    assert c["ts_local"].dt.hour.tolist() == [1, 3]
    # 01:00 AEST and 03:00 AEDT are one real hour apart.
    assert (c["ts_utc"].iloc[1] - c["ts_utc"].iloc[0]) == pd.Timedelta(hours=1)


def test_repeated_autumn_hour_is_read_as_daylight_time():
    # Clocks went from 03:00 back to 02:00 on 6 April 2025; 02:00 happened twice.
    c, dropped = localize(counts_at("2025-04-06 02:00"), TZ)
    assert dropped == 0
    assert c["ts_utc"].iloc[0] == pd.Timestamp(
        "2025-04-05 15:00", tz="UTC"
    )  # 02:00 at UTC+11


def test_utc_offset_follows_daylight_saving():
    c, _ = localize(counts_at("2025-01-15 12:00", "2025-07-15 12:00"), TZ)
    assert c["ts_utc"].dt.hour.tolist() == [1, 2]  # UTC+11 in summer, UTC+10 in winter


def test_duplicate_sensor_hours_are_dropped_entirely():
    c = pd.concat(
        [
            counts_at("2010-09-01 00:00", count=5),
            counts_at("2010-09-01 00:00", count=7),
            counts_at("2010-09-02 00:00"),
        ]
    )
    kept, n = drop_unplaceable(c)
    assert n == 2 and len(kept) == 1


def test_wayback_export_is_read_like_the_live_table(tmp_path):
    # The portal's CSV export is ;-separated and carries the same columns as the live table.
    path = tmp_path / "wayback.csv"
    path.write_text(
        "id;location_id;sensing_date;hourday;direction_1;direction_2;pedestriancount;"
        "sensor_name;location\n"
        "77220230808;77;2023-08-08;2;1;0;1;HarEsP_T;-37.81441438,144.94433026\n"
    )
    w = read_wayback(path)
    assert w["ts_local_naive"].iloc[0] == pd.Timestamp("2023-08-08 02:00")
    assert w[["pedestriancount", "direction_1", "direction_2"]].iloc[0].tolist() == [
        1,
        1,
        0,
    ]
    assert w["source"].iloc[0] == "wayback"


def test_wayback_fills_only_the_gap_between_archive_and_live():
    archive = counts_at("2022-10-31 22:00", "2022-10-31 23:00")
    live = counts_at("2024-10-21 00:00", "2024-10-21 01:00")
    wayback = counts_at(
        "2022-10-31 23:00",  # the archive has this hour
        "2022-11-01 00:00",
        "2023-06-15 12:00",
        "2024-10-20 23:00",
        "2024-10-21 00:00",  # the live table has this hour
    )
    kept = fill_gap(wayback, archive, live)
    assert kept["ts_local_naive"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2022-11-01 00:00",
        "2023-06-15 12:00",
        "2024-10-20 23:00",
    ]


def hourly(sensor, start, counts):
    ts = pd.date_range(start, periods=len(counts), freq="h", tz=TZ)
    return pd.DataFrame(
        {
            "location_id": sensor,
            "ts_local": ts,
            "ts_utc": ts.tz_convert("UTC"),
            "pedestriancount": counts,
        }
    )


def test_outage_flag_needs_a_daytime_zero_run():
    night = hourly(
        1, "2025-03-03 01:00", [0, 0, 0, 0, 5]
    )  # 01:00-04:00: quiet, not broken
    day = hourly(2, "2025-03-03 10:00", [9, 0, 0, 0, 9])  # three zeros at midday
    short = hourly(3, "2025-03-03 10:00", [9, 0, 0, 9, 9])  # only two
    c = pd.concat([night, day, short], ignore_index=True)
    flags = flag_outages(c, min_run=3, day_hours=(8, 20))
    assert not flags[c.location_id == 1].any()
    assert flags[c.location_id == 2].tolist() == [False, True, True, True, False]
    assert not flags[c.location_id == 3].any()


def test_outage_run_breaks_at_a_gap():
    a = hourly(1, "2025-03-03 10:00", [0, 0])
    b = hourly(1, "2025-03-03 15:00", [0])  # three zeros, but not consecutive hours
    c = pd.concat([a, b], ignore_index=True)
    assert not flag_outages(c, min_run=3, day_hours=(8, 20)).any()


@pytest.mark.parametrize(
    "utc, expected",
    [
        # Solar noon in Melbourne is about 02:20 UTC; elevation = 90 - latitude + declination.
        ("2025-12-21 02:20", 90 - 37.8136 + 23.44),
        ("2025-03-20 02:20", 90 - 37.8136),
        ("2025-06-21 02:20", 90 - 37.8136 - 23.44),
    ],
)
def test_solar_elevation_at_noon(utc, expected):
    got = solar_elevation(pd.Series(pd.to_datetime([utc], utc=True)), MELB[1], MELB[0])[
        0
    ]
    assert got == pytest.approx(expected, abs=1.0)


def test_sun_is_down_at_local_midnight():
    got = solar_elevation(
        pd.Series(pd.to_datetime(["2025-06-21 14:00"], utc=True)), MELB[1], MELB[0]
    )
    assert got[0] < -40


def test_weather_is_read_on_its_fixed_offset_and_split_by_variable(tmp_path):
    w = {
        "utc_offset_seconds": 36000,
        "hourly": {
            "time": ["2025-01-15T12:00", "2025-01-15T13:00", "2025-01-15T14:00"],
            "temperature_2m": [20.0, 21.0, 22.0],
            "precipitation": [0.0, 1.5, 3.0],
        },
    }
    (tmp_path / "melbourne_openmeteo_2025.json").write_text(json.dumps(w))
    weather = read_weather(tmp_path, "melbourne")
    # 12:00 at UTC+10 is 02:00 UTC, whatever the local clock says in January.
    assert weather.index[0] == pd.Timestamp("2025-01-15 02:00", tz="UTC")
    # A count for 13:00-14:00 local daylight time is 02:00-03:00 UTC.
    ts = pd.Series(pd.to_datetime(["2025-01-15 02:00"], utc=True))
    wx = weather_for_counts(ts, weather, ["precipitation"])
    assert wx["wx_temperature_2m"].iloc[0] == 20.0  # instant at the start of the hour
    assert wx["wx_precipitation"].iloc[0] == 1.5  # sum stamped at the end of the hour


def test_calendar_flags_are_unknown_outside_covered_years():
    dates = pd.Series(pd.to_datetime(["2014-03-10", "2015-03-10"]))
    flag = pd.Series([True, True])
    out = covered(dates, flag, {2015})
    assert out.isna().tolist() == [True, False]
    assert bool(out.iloc[1]) is True
    assert np.dtype(out.dtype.numpy_dtype) == np.dtype(bool)
