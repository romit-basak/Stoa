"""The NYC transfer test's pieces: count parsing, candidate days, snapping and scores."""

import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.models.flow import transfer_nyc as tn

SNAP = {"search_radius_m": 60, "sidewalk_radius_m": 35, "sidewalk_angle_deg": 30}


def test_parse_counts_reads_every_column_style_and_drops_zeros():
    points = pd.DataFrame(
        {
            "loc": [1, 2],
            "may_22_p_m": [100, 0],
            "oct24_am": [50, 60],
            "june_24_md": [0, 70],
            "sept_09_am": [10, None],
            "street_nam": ["A", "B"],
        }
    )
    out = tn.parse_counts(points)
    got = set(
        out[["loc", "year", "month", "period", "observed"]].itertuples(index=False)
    )
    assert got == {
        (1, 2022, 5, "pm", 100),
        (1, 2024, 10, "am", 50),
        (2, 2024, 10, "am", 60),
        (2, 2024, 6, "md", 70),
        (1, 2009, 9, "am", 10),
    }


def test_count_days_are_saturdays_or_listed_weekdays_without_holidays():
    sats = tn.count_days(2025, 5, "md", [1, 2, 3], set())
    assert sats and all(d.dayofweek == 5 for d in sats)
    assert all(d.month == 5 for d in sats)
    holiday = dt.date(2025, 10, 14)  # a Tuesday
    days = tn.count_days(2025, 10, "am", [1, 2, 3], {holiday})
    assert all(d.dayofweek in (1, 2, 3) for d in days)
    assert holiday not in {d.date() for d in days}
    assert len(days) == 13  # 14 Tuesdays to Thursdays in October 2025, less the holiday


@pytest.mark.parametrize(
    "a,b",
    [
        ("East 42 Street", "East 42nd Street"),
        ("Fifth Avenue", "Fifth Ave"),
        ("West 125th Street", "W 125 St"),
    ],
)
def test_street_names_compare_after_normalising(a, b):
    assert tn.normalise_name(a) == tn.normalise_name(b)


def _segments(with_sidewalks: bool) -> gpd.GeoDataFrame:
    rows = [
        ("road", "Main Street", "secondary", False, LineString([(-100, 0), (100, 0)])),
        ("crossing", None, "footway", False, LineString([(5, -10), (5, 10)])),
        (
            "other",
            "Side Street",
            "residential",
            False,
            LineString([(0, -100), (0, -40)]),
        ),
    ]
    if with_sidewalks:
        rows += [
            ("north", None, "footway", True, LineString([(-100, 8), (100, 8)])),
            ("north_far", None, "footway", True, LineString([(-100, 20), (100, 20)])),
            ("south", None, "footway", True, LineString([(-100, -9), (100, -9)])),
        ]
    df = pd.DataFrame(
        rows, columns=["segment_id", "name", "highway", "is_sidewalk", "geometry"]
    )
    df["is_crossing"] = df["segment_id"].eq("crossing")
    return gpd.GeoDataFrame(df, geometry="geometry", crs=32618)


def _point() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"loc": [7], "street_nam": ["Main St"]}, geometry=[Point(1, 1)], crs=32618
    )


def test_snap_takes_the_nearest_sidewalk_on_each_side_of_the_named_street():
    units = tn.snap(_point(), _segments(True), SNAP)
    assert set(units["segment_id"]) == {"north", "south"}
    assert set(units["kind"]) == {"both_sidewalks"}
    assert units["street_name_matched"].all()


def test_snap_falls_back_to_one_segment_without_sidewalks_on_both_sides():
    units = tn.snap(_point(), _segments(False), SNAP)
    assert len(units) == 1
    assert units.iloc[0]["kind"] == "single_path"
    assert units.iloc[0]["segment_id"] == "road"  # crossings are never picked


def test_window_totals_sum_hours_and_sides_then_average_the_days():
    x = pd.DataFrame(
        {
            "loc": 1,
            "year": 2025,
            "month": 5,
            "period": "am",
            "day": ["d1"] * 4 + ["d2"] * 4,
            "weight": 1.0,
            "predicted": [1, 2, 3, 4, 10, 20, 30, 40],
        }
    )
    out = tn.window_totals(x, ["predicted"])
    assert out["predicted"].item() == pytest.approx((10 + 100) / 2)


def _totals(pred_scale: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    obs = rng.integers(100, 5000, size=40)
    return pd.DataFrame(
        {
            "loc": np.repeat(np.arange(20), 2),
            "year": 2025,
            "month": 5,
            "period": np.tile(["am", "pm"], 20),
            "observed": obs,
            "predicted": (obs + 1) * pred_scale - 1,
        }
    )


def test_scores_are_perfect_for_exact_predictions():
    s = tn.scores(_totals(1.0), "predicted")
    assert s["log_mae"] == pytest.approx(0.0, abs=1e-9)
    assert s["spearman_mean"] == pytest.approx(1.0)


def test_a_city_wide_scale_removes_a_constant_offset():
    t = _totals(0.5)
    s = tn.scores(t, "predicted")
    assert s["median_log_ratio"] == pytest.approx(np.log(0.5))
    assert s["log_mae_offset_removed"] == pytest.approx(0.0, abs=1e-9)
    assert s["spearman_mean"] == pytest.approx(1.0)
    cal = tn.calibrated(t, "predicted", 3, 20, np.random.default_rng(1))
    assert cal["median"] == pytest.approx(0.0, abs=1e-9)
