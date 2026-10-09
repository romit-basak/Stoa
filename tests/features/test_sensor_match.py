import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.features.sensor_match import axis_diff, match_sensors, sensor_axis
from tests.features.conftest import CRS


@pytest.fixture
def street():
    """A road centreline (footpaths mapped separately) with a footway 8 m north of it."""
    return gpd.GeoDataFrame(
        {
            "segment_id": ["road-0", "foot-0", "cross-0"],
            "highway": ["residential", "footway", "footway"],
            "name": ["Test St", None, None],
            "sidewalk_state": ["separate", None, None],
            "is_crossing": [False, False, True],
            "bearing_deg": [90.0, 90.0, 0.0],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 8), (100, 8)]),
            LineString([(52, 0), (52, 8)]),
        ],
        crs=CRS,
    )


def sensors(*rows):
    return gpd.GeoDataFrame(
        [
            {
                "location_id": lid,
                "sensor_description": f"sensor {lid}",
                "location_type": kind,
                "direction_1": d1,
                "direction_2": None,
                "geometry": Point(x, y),
            }
            for lid, (x, y), kind, d1 in rows
        ],
        crs=CRS,
    )


def test_skips_road_with_separate_footpaths(street, fcfg):
    # Closer to the road (3 m) than the footway (5 m), but the road's footpaths are separate.
    m, _ = match_sensors(
        sensors((1, (20, 3), "Outdoor", "East")), street, fcfg["sensor_match"]
    )
    r = m.iloc[0]
    assert r.segment_id == "foot-0" and r.distance_m == pytest.approx(5.0)
    assert r.bearing_agrees
    assert not r.needs_review


def test_crossing_penalty_and_review_flags(street, fcfg):
    # 1.5 m from the crossing and 6 m from the footway: with the 5 m crossing penalty the
    # footway wins, but only by 0.5 m, so a person should check it.
    m, cand = match_sensors(
        sensors((2, (50.5, 2), "Outdoor", "North")), street, fcfg["sensor_match"]
    )
    r = m.iloc[0]
    assert r.segment_id == "foot-0"
    assert r.runner_up_gap_m == pytest.approx(0.5)
    assert "close runner-up" in r.review_reasons
    # A north-south sensor on an east-west path.
    assert "bearing disagrees" in r.review_reasons
    assert len(cand) == 2


def test_far_indoor_and_unmatched(street, fcfg):
    m, _ = match_sensors(
        sensors(
            (3, (20, 26), "Outdoor", None),
            (4, (20, 300), "Outdoor", None),
            (5, (20, 9), "Indoor", None),
        ),
        street,
        fcfg["sensor_match"],
    )
    m = m.set_index("location_id")
    assert "far" in m.loc[3, "review_reasons"]
    assert m.loc[4, "method"] == "none" and pd.isna(m.loc[4, "segment_id"])
    assert "indoor sensor" in m.loc[5, "review_reasons"]
    assert m.needs_review.all()


def test_override_wins(street, fcfg):
    m, _ = match_sensors(
        sensors((1, (20, 3), "Outdoor", None)),
        street,
        fcfg["sensor_match"],
        overrides={1: "road-0"},
    )
    r = m.iloc[0]
    assert (r.segment_id, r.method, r.needs_review) == ("road-0", "override", False)


def test_axes():
    assert sensor_axis("North", "South") == 0.0
    assert sensor_axis("In", "East") == 90.0
    assert sensor_axis("In", "Out") is None
    assert axis_diff(5, 175) == pytest.approx(10)
    assert axis_diff(0, 90) == pytest.approx(90)
