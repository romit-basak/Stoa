from datetime import UTC, datetime

import pandas as pd

from src.pipeline import minute_checks as mc

CFG = {
    "min_last_local": "23:00",
    "max_duplicates": 0,
    "sum_tolerance": 0.05,
    "max_mismatch_share": 0.05,
}
NOW = datetime(2026, 10, 27, 3, 0, tzinfo=UTC)  # 2 pm on 27 Oct in Melbourne
# The hourly table's newest hour is never compared (the portal publishes it unfinished), so
# tables end with this later hour.
LATER = "2026-10-26 22:00"


def minutes(*rows):
    """(sensor, Melbourne local minute, count)."""
    return pd.DataFrame(
        [
            {
                "location_id": lid,
                "sensing_datetime": pd.Timestamp(t, tz=mc.TZ).tz_convert("UTC"),
                "total_of_directions": n,
            }
            for lid, t, n in rows
        ]
    )


def by_name(checks):
    return {c.name: c for c in checks}


def test_a_complete_archive_passes():
    m = minutes(
        (1, "2026-10-25 23:59", 1),
        (1, "2026-10-26 01:00", 2),
        (1, "2026-10-26 01:01", 3),
        (1, "2026-10-26 23:30", 1),
        (1, "2026-10-27 13:00", 1),
    )
    hourly = pd.DataFrame(
        {
            "location_id": [1, 1],
            "ts_local": pd.to_datetime(["2026-10-26 01:00", LATER]).tz_localize(mc.TZ),
            "pedestriancount": [5, 0],
        }
    )
    c = by_name(mc.minute_checks(m, hourly, CFG, NOW))
    assert all(x.passed for x in c.values()), [
        x.name for x in c.values() if not x.passed
    ]
    assert (
        c["minute feed: minutes add up to the hourly counts"].observed["compared"] >= 1
    )


def test_a_missing_day_and_an_early_last_poll_warn():
    # 25 Oct ends at 21:00 (no poll after that), 26 Oct has no file at all.
    m = minutes((1, "2026-10-25 21:00", 1), (1, "2026-10-27 13:00", 1))
    c = by_name(
        mc.minute_checks(
            m,
            pd.DataFrame(columns=["location_id", "ts_local", "pedestriancount"]),
            CFG,
            NOW,
        )
    )
    assert c["minute feed: a file for every day"].observed["missing"] == ["2026-10-26"]
    assert c["minute feed: polled late enough each day"].observed["days"] == [
        "2026-10-25"
    ]


def test_minutes_that_do_not_add_up_warn():
    m = minutes(
        (1, "2026-10-26 10:00", 10),
        (1, "2026-10-26 10:30", 10),
        (1, "2026-10-26 23:30", 1),
    )
    hourly = pd.DataFrame(
        {
            "location_id": [1, 1],
            "ts_local": pd.to_datetime(["2026-10-26 10:00", LATER]).tz_localize(mc.TZ),
            "pedestriancount": [50, 0],
        }
    )
    c = by_name(mc.minute_checks(m, hourly, CFG, NOW))
    assert not c["minute feed: minutes add up to the hourly counts"].passed


def test_duplicates_are_counted():
    m = minutes(
        (1, "2026-10-26 10:00", 4),
        (1, "2026-10-26 10:00", 26),
        (1, "2026-10-26 23:30", 1),
    )
    c = by_name(
        mc.minute_checks(
            m,
            pd.DataFrame(columns=["location_id", "ts_local", "pedestriancount"]),
            CFG,
            NOW,
        )
    )
    assert c["minute feed: one row per sensor and minute"].observed["duplicates"] == 1


def test_no_archive_yet_passes():
    (c,) = mc.minute_checks(pd.DataFrame(), pd.DataFrame(), CFG, NOW)
    assert c.passed and "no archive" in c.detail


def test_the_hourly_tables_unfinished_newest_hour_is_not_compared():
    m = minutes((1, "2026-10-26 10:00", 40), (1, "2026-10-26 23:30", 1))
    # The portal published 10:00 when only 12 people had been counted.
    hourly = pd.DataFrame(
        {
            "location_id": [1],
            "ts_local": [pd.Timestamp("2026-10-26 10:00", tz=mc.TZ)],
            "pedestriancount": [12],
        }
    )
    c = by_name(mc.minute_checks(m, hourly, CFG, NOW))
    check = c["minute feed: minutes add up to the hourly counts"]
    assert check.passed and check.observed["compared"] == 0


def test_an_archive_that_began_mid_hour_doesnt_compare_that_hour():
    # The archive began at 00:55, so its 00:00 hour holds 5 minutes of 60.
    m = minutes(
        (1, "2026-10-26 00:55", 3),
        (1, "2026-10-26 01:10", 40),
        (1, "2026-10-26 23:30", 1),
    )
    hourly = pd.DataFrame(
        {
            "location_id": [1, 1, 1],
            "ts_local": pd.to_datetime(
                ["2026-10-26 00:00", "2026-10-26 01:00", LATER]
            ).tz_localize(mc.TZ),
            "pedestriancount": [30, 40, 0],
        }
    )
    c = by_name(mc.minute_checks(m, hourly, CFG, NOW))
    check = c["minute feed: minutes add up to the hourly counts"]
    assert check.passed and check.observed["compared"] == 1
