import pandas as pd
import pytest

from src.bias.minute_coverage import coverage

CFG = {"low_ratio": 0.5, "min_hours": 4}


def status_and_slices():
    # 8 busy daytime hours, all kept; 8 night hours, 1 kept (the rest too quiet).
    status = pd.DataFrame(
        {
            "location_id": [1, 2] * 8,
            "in_archive": [True] * 16,
            "adds_up": [True] * 16,
            "kept": [True] * 8 + [True] + [False] * 7,
        }
    )
    slices = pd.DataFrame({"time_of_day": ["midday"] * 8 + ["night"] * 8})
    return status, slices


def test_a_group_the_profiles_mostly_drop_is_flagged_under():
    status, slices = status_and_slices()
    t = coverage(status, slices, ["time_of_day"], CFG).set_index("group")
    assert t.loc["midday", "kept_share"] == pytest.approx(1.0)
    assert t.loc["night", "kept_share"] == pytest.approx(1 / 8)
    # Overall 9/16 kept: night's ratio is (1/8) / (9/16) = 0.22, under 0.5.
    assert bool(t.loc["night", "under"]) and not bool(t.loc["midday", "under"])
    assert t.loc["night", "sensors"] == 2


def test_small_groups_are_reported_but_not_flagged():
    status, slices = status_and_slices()
    t = coverage(status, slices, ["time_of_day"], {**CFG, "min_hours": 100})
    assert not t["under"].any() and len(t) == 2


def test_no_archive_yet_gives_an_empty_table():
    empty = pd.DataFrame(columns=["location_id", "in_archive", "adds_up", "kept"])
    assert coverage(empty, pd.DataFrame(), [], CFG).empty


def test_pooled_profiles_sum_quiet_hours_until_they_are_usable():
    import numpy as np

    from src.bias.minute_coverage import pooled_profiles

    # Three quiet hours of 10 people each: too few alone, 30 together.
    summary = pd.DataFrame(
        {
            "location_id": [1, 2, 3],
            "volume_tier": ["quiet"] * 3,
            "day_type": ["weekday"] * 3,
            "hour_local": [2, 2, 2],
        }
    )
    grid = np.zeros((3, 60))
    grid[:, 0] = 10
    p = pooled_profiles(summary, grid, min_people=30)
    cell = p[p["minute"] == 0].iloc[0]
    assert cell["people"] == 30 and cell["n_sensors"] == 3 and bool(cell["usable"])
    assert cell["share"] == pytest.approx(1.0)
    assert p["share"].sum() == pytest.approx(1.0)
    assert not pooled_profiles(summary, grid, min_people=31)["usable"].any()
