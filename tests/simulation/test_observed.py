import pandas as pd
import pytest

from src.simulation import observed

AREA = {
    "sensors": [1, 2],
    "lead_sensor": 1,
    "event_venue": "arena",
    "event_kinds": ["concert"],
    "nearby_venues": ["arena", "court"],
    "window_from_hours": 2.5,
    "window_to_hours": 3.0,
    "block_minutes": 5,
}


def event(name, kind, venue, start):
    return {"name": name, "kind": kind, "venue": venue, "start": start}


def local(text):
    return pd.Timestamp(text, tz=observed.TZ)


def feed_rows(sensor, local_times, totals, polled="2026-10-27 12:00"):
    utc = pd.to_datetime(local_times).tz_localize(observed.TZ).tz_convert("UTC")
    return pd.DataFrame(
        {
            "location_id": sensor,
            "sensing_datetime": utc,
            "direction_1": totals,
            "direction_2": [0] * len(totals),
            "total_of_directions": totals,
            "polled_at": pd.Timestamp(polled, tz="UTC"),
        }
    )


def test_events_come_from_any_list_with_venues_and_get_local_times():
    raw = {
        "venues": {"arena": {"name": "Arena"}},
        "events": [
            event("A", "concert", "arena", "2026-10-27T19:30"),
            event("B", "netball", "court", "2026-10-27T19:00"),
        ],
    }
    events = observed.event_table(raw)
    assert list(events["name"]) == ["B", "A"]
    assert str(events["start"].dt.tz) == observed.TZ


def test_area_events_keep_the_venue_and_kind_and_flag_other_events():
    raw = {
        "events": [
            event("A", "concert", "arena", "2026-10-27T19:30"),
            event("B", "netball", "court", "2026-10-27T19:00"),
            event("C", "concert", "arena", "2026-10-28T19:30"),
        ]
    }
    events = observed.area_events(observed.event_table(raw), AREA)
    assert list(events["name"]) == ["A", "C"]
    assert list(events["other_events_that_night"]) == [1, 0]


def test_minute_counts_keep_the_latest_poll_and_use_local_time():
    first = feed_rows(1, ["2026-10-27 22:00"], [5], polled="2026-10-27 11:00")
    later = feed_rows(1, ["2026-10-27 22:00"], [9], polled="2026-10-27 11:30")
    other = feed_rows(3, ["2026-10-27 22:00"], [7])
    counts = observed.minute_counts(pd.concat([first, later, other]), [1, 2])
    assert len(counts) == 1
    assert counts["total"].iloc[0] == 9
    assert counts["ts_local"].iloc[0] == local("2026-10-27 22:00")


def test_blocks_sum_minutes_into_five_minute_blocks():
    rows = feed_rows(1, [f"2026-10-27 22:0{m}" for m in range(7)], [1] * 7)
    b = observed.blocks(observed.minute_counts(rows, [1]), 5)
    assert list(b["total"]) == [5, 2]


def test_series_cover_only_the_egress_window_and_summary_finds_peaks_and_delays():
    times = pd.date_range("2026-10-27 21:50", "2026-10-27 22:40", freq="5min")
    lead = [1, 1, 1, 9, 30, 10, 2, 1, 1, 1, 1]
    next_ = [0, 0, 0, 1, 5, 20, 4, 1, 0, 0, 0]
    feed = pd.concat(
        [
            feed_rows(1, times.strftime("%Y-%m-%d %H:%M"), lead),
            feed_rows(2, times.strftime("%Y-%m-%d %H:%M"), next_),
        ]
    )
    raw = {"events": [event("A", "concert", "arena", "2026-10-27T19:30")]}
    events = observed.area_events(observed.event_table(raw), AREA)
    counts = observed.blocks(observed.minute_counts(feed, [1, 2]), 5)
    series = observed.egress_series(counts, events, AREA)
    assert series["block"].min() == local("2026-10-27 22:00")
    assert series["block"].max() == local("2026-10-27 22:25")
    summary = observed.summarise(series, lead_sensor=1).set_index("location_id")
    assert summary.loc[1, "peak_count"] == 30
    assert summary.loc[1, "peak_block"] == local("2026-10-27 22:10")
    assert summary.loc[2, "peak_delay_min"] == pytest.approx(5.0)
    assert summary.loc[1, "blocks"] == 6


def test_a_night_without_counts_gives_empty_series():
    raw = {"events": [event("A", "concert", "arena", "2026-11-12T19:30")]}
    events = observed.area_events(observed.event_table(raw), AREA)
    counts = observed.blocks(
        observed.minute_counts(feed_rows(1, ["2026-10-27 22:00"], [3]), [1]), 5
    )
    assert observed.egress_series(counts, events, AREA).empty
