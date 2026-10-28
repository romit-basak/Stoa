from datetime import UTC, datetime

import pandas as pd

from src.pipeline import minute_checks as mc

CFG = {
    "min_last_local": "23:00",
    "max_duplicates": 0,
    "sum_tolerance": 0.05,
    "max_mismatch_share": 0.05,
    "max_gappy_sensors": 0,
}
REPORTING = {"five_minute_share": 0.99, "gap_min_rows": 300}
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


def test_a_minute_revised_by_a_later_poll_is_not_a_duplicate_and_its_latest_value_counts():
    first = pd.Timestamp("2026-10-26 00:00", tz="UTC")
    m = pd.concat(
        [
            minutes((1, "2026-10-26 10:00", 4)).assign(polled_at=first),
            minutes((1, "2026-10-26 10:00", 26)).assign(
                polled_at=first + pd.Timedelta(minutes=15)
            ),
            minutes((1, "2026-10-26 23:30", 1)).assign(
                polled_at=first + pd.Timedelta(hours=14)
            ),
        ]
    )
    hourly = pd.DataFrame(
        {
            "location_id": [1, 1],
            "ts_local": pd.to_datetime(["2026-10-26 10:00", LATER]).tz_localize(mc.TZ),
            "pedestriancount": [26, 0],
        }
    )
    c = by_name(mc.minute_checks(m, hourly, CFG, NOW))
    dup = c["minute feed: one row per sensor and minute"]
    assert dup.passed and dup.observed == {"duplicates": 0, "revised": 1}
    sums = c["minute feed: minutes add up to the hourly counts"]
    assert sums.passed and sums.observed["compared"] == 1


def test_sensors_missing_a_block_warn_and_are_left_out_of_the_sums():
    # Sensor 1 never publishes minutes :00-:04, so its hours run short of the hourly table.
    hours = pd.date_range("2026-10-26 00:00", periods=8, freq="h")
    rows = [
        (1, str(h + pd.Timedelta(minutes=k)), 3) for h in hours for k in range(5, 60)
    ]
    rows += [(2, str(h), 10) for h in hours]
    m = minutes(*rows)
    hourly = pd.DataFrame(
        {
            "location_id": [1] * 8 + [2] * 8 + [2],
            "ts_local": [*hours, *hours, pd.Timestamp(LATER)],
            "pedestriancount": [60 * 3] * 8 + [10] * 8 + [0],
        }
    )
    hourly["ts_local"] = pd.to_datetime(hourly["ts_local"]).dt.tz_localize(mc.TZ)
    c = by_name(mc.minute_checks(m, hourly, CFG, NOW, REPORTING))
    gap = c["minute feed: sensors publish every 5-minute block"]
    assert not gap.passed and gap.observed["sensors"] == [1] and "(0-4)" in gap.detail
    sums = c["minute feed: minutes add up to the hourly counts"]
    # Only sensor 2's hours are compared (the newest of them is the archive's newest, left out).
    assert sums.passed and sums.observed["compared"] == 7
