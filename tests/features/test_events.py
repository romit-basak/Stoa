import pandas as pd
import pytest

from src.features import events as ev
from src.features.common import load_features_config

CFG = load_features_config()["events"]
LOG = {
    "events": [
        {"kind": "cricket", "venue": "mcg", "start": "2026-11-11T10:30", "end": None},
        {
            "kind": "cricket",
            "venue": "mcg",
            "start": "2026-11-21T19:15",
            "end": None,
            "duration_h": 3.5,
        },
        {
            "kind": "concert",
            "venue": "marvel",
            "start": "2026-11-14T00:00",
            "end": None,
            "time_note": "date only",
        },
        {
            "kind": "halloween",
            "venue": None,
            "start": "2026-10-31T00:00",
            "end": "2026-11-01T00:00",
            "scope": "citywide",
        },
    ]
}
AFL = pd.DataFrame(
    {
        "date": ["2026-04-03", "2026-04-04"],
        "local_time": ["7:40 PM", "1:10 PM"],
        "venue": ["M.C.G.", "Gabba"],
    }
)
VENUES = {"M.C.G.": "mcg", "Docklands": "marvel"}


def test_timed_events_take_the_kind_default_or_their_own_length():
    t = ev.timed_events(LOG, AFL, VENUES, CFG).set_index("start")
    shield = t.loc[pd.Timestamp("2026-11-11 10:30")]
    assert shield["end"] == pd.Timestamp("2026-11-11 10:30") + pd.Timedelta(
        hours=CFG["default_hours"]["cricket"]
    )
    assert t.loc[pd.Timestamp("2026-11-21 19:15"), "end"] == pd.Timestamp(
        "2026-11-21 22:45"
    )
    afl = t.loc[pd.Timestamp("2026-04-03 19:40")]
    assert (afl["venue"], afl["kind"]) == ("mcg", "afl")
    # Date-only and venue-less events have no hours; the Gabba isn't a Melbourne venue.
    assert len(t) == 3


def test_event_hours_cover_before_during_and_after():
    timed = pd.DataFrame(
        {
            "venue": ["mcg"],
            "kind": ["afl"],
            "start": [pd.Timestamp("2026-04-03 19:40")],
            "end": [pd.Timestamp("2026-04-03 22:40")],
        }
    )
    h = ev.venue_hours(timed, CFG)
    by = {
        p: h.loc[h["phase"] == p, "hour"].sort_values().dt.hour.tolist()
        for p in ev.PHASES
    }
    assert by["pre"] == [17, 18, 19]  # 17:40 to 19:40
    assert by["during"] == [19, 20, 21, 22]
    assert by["post"] == [22, 23, 0]  # 22:40 to 00:40


def test_rows_near_a_venue_get_the_phase_and_the_distance():
    hours = pd.DataFrame(
        {
            "hour": [pd.Timestamp("2026-04-03 20:00")],
            "venue": ["mcg"],
            "phase": ["during"],
        }
    )
    days = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-03"), pd.Timestamp("2026-11-14")],
            "venue": ["mcg", "marvel"],
        }
    )
    dist = pd.DataFrame(
        {
            "segment_id": ["near", "marvel_side"],
            "venue": ["mcg", "marvel"],
            "dist_m": [400.0, 250.0],
        }
    )
    rows = pd.DataFrame(
        {
            "hour": pd.to_datetime(
                [
                    "2026-04-03 20:00",
                    "2026-04-03 20:00",
                    "2026-04-03 09:00",
                    "2026-11-14 12:00",
                ]
            ),
            "segment": ["near", "far", "near", "marvel_side"],
        }
    )
    out = ev.hourly_event_features(
        rows["hour"], rows["segment"], hours, days, dist, cap=3000.0
    )
    assert out["event_during"].tolist() == [True, False, False, False]
    assert out["event_dist_m"].tolist() == [400.0, 3000.0, 3000.0, 3000.0]
    # The whole day counts for the day column, and the date-only concert counts too.
    assert out["event_day_dist_m"].tolist() == [400.0, 3000.0, 400.0, 250.0]


@pytest.mark.parametrize("kind", ["afl", "cricket", "football", "concert", "halloween"])
def test_every_kind_in_the_log_has_a_default_length(kind):
    assert CFG["default_hours"][kind] > 0
