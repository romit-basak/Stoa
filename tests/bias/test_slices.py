import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src.bias.representation import coverage_by_year, crosstab, summarize
from src.bias.slices import (
    UNKNOWN,
    afl_days,
    bins,
    era,
    flag_label,
    load_bias_config,
    missing_hours_share,
    sensor_precincts,
    tiers,
    time_of_day,
)

CFG = load_bias_config()


def test_bins_are_left_closed_with_an_open_last_bin_and_unknown_for_missing():
    rain = pd.Series([0.0, 0.09, 0.1, 0.99, 1.0, 40.0, np.nan])
    assert bins(rain, CFG["rain_mm"]).tolist() == [
        "dry",
        "dry",
        "light",
        "light",
        "rain",
        "rain",
        UNKNOWN,
    ]


def test_time_of_day_boundaries():
    hours = pd.Series([0, 5, 6, 9, 10, 15, 16, 19, 20, 23])
    assert time_of_day(hours, CFG["time_of_day"]).tolist() == [
        "night",
        "night",
        "morning_peak",
        "morning_peak",
        "midday",
        "midday",
        "evening_peak",
        "evening_peak",
        "late_evening",
        "late_evening",
    ]


def test_eras_start_on_their_dates():
    dates = pd.Series(
        pd.to_datetime(["2020-03-15", "2020-03-16", "2021-10-22", "2025-11-30"])
    )
    assert era(dates, CFG["eras"]).tolist() == [
        "pre_covid",
        "covid",
        "recovery",
        "post_metro_tunnel",
    ]


def test_flag_label_keeps_missing_as_unknown():
    flag = pd.Series([True, False, None], dtype="boolean")
    assert flag_label(flag, "term", "holidays").tolist() == [
        "term",
        "holidays",
        UNKNOWN,
    ]


def test_afl_days_by_stadium():
    games = pd.DataFrame(
        {
            "date": ["2025-04-05", "2025-04-05", "2025-04-06", "2025-04-07"],
            "venue": ["Docklands", "M.C.G.", "M.C.G.", "Gabba"],
        }
    )
    days = afl_days(games, CFG["afl_venues"])
    assert days.to_dict() == {
        pd.Timestamp("2025-04-05"): "both",
        pd.Timestamp("2025-04-06"): "mcg",
    }


def test_missing_hours_are_counted_within_each_source_span():
    ts = pd.to_datetime(
        # archive: 10-hour span with 2 hours absent; live: 4 hours, none absent
        ["2022-01-01 00:00", "2022-01-01 01:00", "2022-01-01 04:00"]
        + [f"2022-01-01 0{h}:00" for h in range(5, 10)]
        + [f"2025-01-01 0{h}:00" for h in range(4)],
        utc=True,
    )
    hourly = pd.DataFrame(
        {
            "location_id": 1,
            "source": ["archive"] * 8 + ["live"] * 4,
            "ts_utc": ts,
        }
    )
    # The two-year gap between the sources doesn't count: 2 missing of 14 spanned hours.
    assert missing_hours_share(hourly).loc[1] == pytest.approx(2 / 14)


def test_tiers_are_equal_sized():
    values = pd.Series([5, 1, 9, 3, 7, 2])
    assert tiers(values, ["low", "mid", "high"]).tolist() == [
        "mid",
        "low",
        "high",
        "mid",
        "high",
        "low",
    ]


def test_precincts_merge_cbd_areas_and_take_held_out_sensors_from_the_list():
    blocks = gpd.GeoDataFrame(
        {"area_name": ["CBD East", "CBD West", "Carlton", "Kensington"]},
        geometry=[
            box(0, 0, 10, 10),
            box(10, 0, 20, 10),
            box(0, 10, 10, 20),
            box(10, 10, 20, 20),
        ],
        crs=28355,
    )
    sensors = gpd.GeoDataFrame(
        {"location_id": [1, 2, 31, 188, 85, 9]},
        geometry=[
            Point(5, 5),
            Point(15, 5),
            Point(5, 15),
            Point(6, 16),
            Point(15, 15),
            Point(50, 50),
        ],
        crs=28355,
    )
    out = sensor_precincts(sensors, blocks, CFG).set_index("location_id")
    assert out["precinct"].to_dict() == {
        1: "CBD",
        2: "CBD",
        31: "Carlton",
        188: "Carlton",
        85: "Kensington",
        9: "outside_clue",
    }
    # Held-out precincts follow the sensor lists, not the CLUE areas: 188 is in the Carlton
    # area but not the Carlton test set, and 85 (Macaulay Rd) is in Kensington but held out.
    assert out["holdout_precinct"].to_dict() == {
        1: "training_area",
        2: "training_area",
        31: "carlton",
        188: "training_area",
        85: "north_melbourne",
        9: "training_area",
    }


def test_summarize_shares_ratio_and_thin_flag():
    rows = pd.DataFrame(
        {
            "location_id": [1, 1, 2, 3, 4, 5, 6, 6],
            "rain": ["dry", "dry", "dry", "dry", "rain", "rain", UNKNOWN, "dry"],
            "pedestriancount": [10, 0, 30, 40, 50, 60, 70, 80],
            "outage": [False, True, False, False, False, False, False, False],
        }
    )
    # Reference: rain is 50% of hours, but only 2 of the 6 kept rows with a known value.
    reference = pd.Series({"dry": 50, "rain": 50, UNKNOWN: 999})
    cfg = {"thin_min_sensors": 2, "thin_ratio_below": 0.7}
    out = summarize(rows, "rain", reference, by_sensor=False, cfg=cfg).set_index(
        "group"
    )

    assert out.loc["dry", "n_rows"] == 4  # the outage row is dropped
    assert out.loc["dry", "outage_share"] == pytest.approx(1 / 5)
    assert out["row_share"].sum() == pytest.approx(1)
    assert out.loc["rain", "ratio"] == pytest.approx((2 / 6) / 0.5)
    assert out.loc["dry", "ratio"] == pytest.approx((4 / 6) / 0.5)
    assert np.isnan(out.loc[UNKNOWN, "ratio"])
    assert out.loc["rain", "thin"]  # ratio 0.67 < 0.7
    assert out.loc[UNKNOWN, "thin"]  # one sensor < 2
    assert not out.loc["dry", "thin"]


def test_summarize_reports_the_count_distribution_without_outage_rows():
    rows = pd.DataFrame(
        {
            "location_id": [1, 1, 1, 1, 2],
            "rain": ["dry"] * 5,
            "pedestriancount": [0, 10, 20, 0, 30],
            "outage": [False, False, False, True, False],
        }
    )
    cfg = {"thin_min_sensors": 1, "thin_ratio_below": 0.5}
    out = summarize(rows, "rain", None, by_sensor=False, cfg=cfg).iloc[0]
    assert out["median_count"] == 15  # of 0, 10, 20, 30: the outage zero is left out
    assert out["zero_share"] == pytest.approx(1 / 4)
    assert out["p90_count"] == pytest.approx(27)


def test_crosstab_shares_are_of_all_kept_rows():
    kept = pd.DataFrame(
        {
            "time_of_day": ["night", "night", "midday", "midday"],
            "rain": ["dry", "rain", "dry", "dry"],
            "pedestriancount": [5, 3, 100, 200],
        }
    )
    out = crosstab(kept, "time_of_day", "rain").set_index(["group_a", "group_b"])
    assert out.loc[("midday", "dry"), "n_rows"] == 2
    assert out.loc[("midday", "dry"), "median_count"] == 150
    assert out["row_share"].sum() == pytest.approx(1)
    assert out["dimension_a"].iloc[0] == "time_of_day"


def test_coverage_by_year_counts_sensors_per_year():
    kept = pd.DataFrame(
        {"location_id": [1, 1, 2, 3], "pedestriancount": [1, 2, 3, 4]},
        index=[0, 1, 2, 5],
    )
    years = pd.Series([2009, 2009, 2010, 2011, 2011, 2010], index=range(6))
    out = coverage_by_year(kept, years).set_index("year")
    assert out["n_sensors"].to_dict() == {2009: 1, 2010: 2}
    assert out.loc[2010, "pedestrians"] == 7
