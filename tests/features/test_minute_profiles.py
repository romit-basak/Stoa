import numpy as np
import pandas as pd
import pytest

from src.features import minute_profiles as mp

CFG = {"sum_tolerance": 0.05, "min_hour_count": 30}
TZ = "Australia/Melbourne"


def feed(sensor: int, start_local: str, counts: dict[int, int]) -> pd.DataFrame:
    """Feed rows for one sensor-hour: minute -> count (absent minutes left out, as the feed does)."""
    t0 = pd.Timestamp(start_local, tz=TZ).tz_convert("UTC")
    return pd.DataFrame(
        {
            "location_id": sensor,
            "sensing_datetime": [t0 + pd.Timedelta(minutes=k) for k in counts],
            "total_of_directions": list(counts.values()),
        }
    )


def hourly(rows: list[tuple[int, str, int]]) -> pd.DataFrame:
    # A later hour at another sensor ends the table, since the table's newest hour is never
    # used (the portal publishes it unfinished).
    rows = [*rows, (999, "2026-10-27 00:00", 0)]
    return pd.DataFrame(
        {
            "location_id": [r[0] for r in rows],
            "ts_local": [pd.Timestamp(r[1], tz=TZ) for r in rows],
            "pedestriancount": [r[2] for r in rows],
        }
    )


def tail_hour(sensor: int) -> pd.DataFrame:
    # A later hour, so the hours under test aren't the archive's newest (which is left out).
    return feed(sensor, "2026-10-25 23:00", {0: 1})


def test_an_hour_is_kept_when_its_minutes_add_up_and_absent_minutes_are_zeros():
    minutes = pd.concat([feed(1, "2026-10-25 12:00", {0: 40, 30: 20}), tail_hour(1)])
    hours, grid = mp.minute_grid(minutes, hourly([(1, "2026-10-25 12:00", 60)]), CFG)
    assert len(hours) == 1 and hours.loc[0, "count"] == 60
    assert grid.shape == (1, 60)
    assert grid[0, 0] == 40 and grid[0, 30] == 20 and grid[0].sum() == 60


def test_hours_that_dont_add_up_or_are_quiet_or_newest_are_left_out():
    minutes = pd.concat(
        [
            feed(1, "2026-10-25 12:00", {0: 40}),  # hourly says 60: a missed reading
            feed(2, "2026-10-25 12:00", {5: 10}),  # adds up, but only 10 people
            feed(3, "2026-10-25 23:00", {0: 50}),  # the newest hour
        ]
    )
    table = hourly(
        [
            (1, "2026-10-25 12:00", 60),
            (2, "2026-10-25 12:00", 10),
            (3, "2026-10-25 23:00", 50),
        ]
    )
    hours, _ = mp.minute_grid(minutes, table, CFG)
    assert hours.empty


def test_a_duplicated_minute_counts_once_with_the_latest_poll():
    a = feed(1, "2026-10-25 12:00", {0: 50})
    b = a.assign(total_of_directions=60)
    minutes = pd.concat(
        [
            a.assign(polled_at=pd.Timestamp("2026-10-25 02:00", tz="UTC")),
            b.assign(polled_at=pd.Timestamp("2026-10-25 03:00", tz="UTC")),
            tail_hour(1).assign(polled_at=pd.Timestamp("2026-10-25 13:00", tz="UTC")),
        ]
    )
    hours, grid = mp.minute_grid(minutes, hourly([(1, "2026-10-25 12:00", 60)]), CFG)
    assert grid[0, 0] == 60 and len(hours) == 1


def test_the_repeated_hour_when_daylight_saving_ends_stays_two_hours():
    # 5 April 2026: 02:00-03:00 AEDT is followed by 02:00-03:00 AEST.
    first = pd.Timestamp("2026-04-04 15:00", tz="UTC").tz_convert(TZ)  # 02:00 AEDT
    second = pd.Timestamp("2026-04-04 16:00", tz="UTC").tz_convert(TZ)  # 02:00 AEST
    minutes = pd.DataFrame(
        {
            "location_id": 1,
            "sensing_datetime": [
                first.tz_convert("UTC"),
                second.tz_convert("UTC"),
                (second + pd.Timedelta(hours=3)).tz_convert("UTC"),
            ],
            "total_of_directions": [40, 50, 1],
        }
    )
    table = pd.DataFrame(
        {
            "location_id": 1,
            "ts_local": [first, second, second + pd.Timedelta(hours=5)],
            "pedestriancount": [40, 50, 0],
        }
    )
    hours, _ = mp.minute_grid(minutes, table, CFG)
    assert sorted(hours["count"]) == [40, 50]


def test_peak15_is_one_for_an_even_hour_and_four_for_one_busy_quarter():
    hours = pd.DataFrame(
        {
            "location_id": [1, 1],
            "ts_local": pd.to_datetime(
                ["2026-10-24 12:00", "2026-10-26 12:00"]
            ).tz_localize(TZ),
            "count": [60, 60],
        }
    )
    even = np.ones(60)
    burst = np.zeros(60)
    burst[:15] = 4
    out = mp.hour_summary(hours, np.vstack([even, burst]))
    assert out["peak15"].tolist() == pytest.approx([1.0, 4.0])
    assert out["day_type"].tolist() == ["weekend", "weekday"]


def test_profiles_average_the_shares_by_group():
    summary = pd.DataFrame(
        {
            "location_id": [1, 1],
            "day_type": ["weekday", "weekday"],
            "hour_local": [12, 12],
        }
    )
    a, b = np.zeros(60), np.zeros(60)
    a[0], b[1] = 30, 60
    prof = mp.profiles(summary, np.vstack([a, b]))
    assert len(prof) == 60 and (prof["n_hours"] == 2).all()
    share = prof.set_index("minute")["share"]
    assert share[0] == pytest.approx(0.5) and share[1] == pytest.approx(0.5)
    assert share.sum() == pytest.approx(1.0)


def test_hour_status_says_what_became_of_each_hour():
    minutes = pd.concat(
        [
            feed(1, "2026-10-25 12:00", {0: 40, 1: 20}),  # adds up to 60: kept
            feed(2, "2026-10-25 12:00", {0: 10}),  # adds up, but quiet
            feed(3, "2026-10-25 12:00", {0: 5}),  # hourly says 50: a missed reading
            tail_hour(1),
        ]
    )
    table = hourly(
        [
            (1, "2026-10-25 12:00", 60),
            (2, "2026-10-25 12:00", 10),
            (3, "2026-10-25 12:00", 50),
            (4, "2026-10-25 12:00", 70),  # no minutes in the archive at all
            (5, "2026-10-25 12:00", 0),  # nobody counted, so no rows: still covered
            (1, "2026-10-24 12:00", 70),  # before the archive began
        ]
    )
    s = mp.hour_status(mp.archive_minutes(minutes), table, CFG).set_index("location_id")
    assert len(s) == 5
    assert s.loc[1, ["in_archive", "adds_up", "kept"]].tolist() == [True, True, True]
    assert s.loc[2, ["in_archive", "adds_up", "kept"]].tolist() == [True, True, False]
    assert s.loc[3, ["in_archive", "adds_up", "kept"]].tolist() == [True, False, False]
    assert s.loc[4, ["in_archive", "adds_up", "kept"]].tolist() == [False, False, False]
    assert s.loc[5, ["in_archive", "adds_up", "kept"]].tolist() == [True, True, False]


def test_no_archive_files_give_empty_results_not_an_error(tmp_path):
    # A clone without bucket credentials: pull_minute_feed is skipped and the folder is empty.
    minutes = mp.read_archive(tmp_path)
    table = hourly([(1, "2026-10-25 12:00", 60)])
    assert mp.hour_status(mp.archive_minutes(minutes), table, CFG).empty
    hours, grid = mp.minute_grid(minutes, table, CFG)
    assert hours.empty and grid.shape == (0, 60)
    # The rest of what main() runs.
    assert mp.profiles(mp.hour_summary(hours, grid), grid).empty
