import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from src.features import shade as sh
from src.features.common import load_features_config

CFG = load_features_config()["shade"]
CRS = 28355


def one_building(height=10.0):
    """A 10 m square, height m tall, at the origin."""
    return gpd.GeoDataFrame(
        {"height_m": [height]}, geometry=[box(0, 0, 10, 10)], crs=CRS
    )


def segments(*lines):
    return gpd.GeoDataFrame(
        {"segment_id": [f"s{i}" for i in range(len(lines))]},
        geometry=[LineString(c) for c in lines],
        crs=CRS,
    )


def test_a_box_casts_its_shadow_away_from_the_sun_as_far_as_height_over_tan():
    # Sun due south at 45 degrees (azimuth 180): the shadow reaches 10 m north of the box.
    shadow = sh.shadows(one_building(10), azimuth=180, elevation=45)[0]
    assert shadow.bounds == pytest.approx((0, 0, 10, 20))
    # At 30 degrees it reaches height / tan(30) = 17.3 m.
    low = sh.shadows(one_building(10), azimuth=180, elevation=30)[0]
    assert low.bounds[3] == pytest.approx(10 + 10 / np.tan(np.radians(30)))


def test_shade_is_the_share_of_a_segment_in_shadow():
    edges = segments(
        [(-5, 15), (15, 15)],  # crosses the shadow north of the box for 10 of its 20 m
        [(0, 30), (10, 30)],  # beyond the shadow
    )
    points, rows = sh.sample_points(edges, 1.0)
    frac = sh.shaded_fraction(
        points, rows, len(edges), sh.shadows(one_building(), 180, 45)
    )
    assert frac[0] == pytest.approx(0.5, abs=0.06)
    assert frac[1] == 0.0


def test_the_same_sun_from_the_north_shades_the_south_side():
    # Melbourne's noon sun is north (azimuth 0): the shadow falls south of the box.
    shadow = sh.shadows(one_building(10), azimuth=0, elevation=45)[0]
    assert shadow.bounds == pytest.approx((0, -10, 10, 10))


def test_sun_bins_round_trip_and_low_sun_counts_as_down():
    b = sh.bin_of(np.array([12.0, 3.0]), np.array([200.0, 90.0]), CFG)
    assert b[1] == -1  # below 5 degrees
    az, el = sh.bin_centre(b[0], CFG)
    assert 180 <= az < 210 and 10 <= el < 20


def test_the_sun_reaches_different_bins_in_each_hemisphere():
    high_south = {
        b
        for b in range(200)
        if sh.bin_centre(b, CFG)[1] >= 45 and 150 <= sh.bin_centre(b, CFG)[0] < 210
    }
    boston = set(sh.reachable_bins(42.34, -71.09, CFG))
    melbourne = set(sh.reachable_bins(-37.81, 144.96, CFG))
    assert high_south & boston  # a high southern sun in Boston
    assert not high_south & melbourne  # never in Melbourne


def test_osm_heights_fall_back_from_tag_to_levels_to_the_default():
    b = gpd.GeoDataFrame(
        {"height": ["12 m", None, None], "building_levels": [None, "4", None]},
        geometry=[box(0, 0, 1, 1)] * 3,
        crs=CRS,
    )
    out = sh.osm_heights(b, CFG)
    assert out["height_m"].tolist() == [12.0, 12.0, CFG["default_height_m"]]
    assert out["height_source"].tolist() == ["tag", "levels", "default"]


def test_the_table_keeps_only_shaded_rows_and_both_variants():
    edges = segments([(-5, 15), (15, 15)], [(100, 100), (110, 100)])
    table = sh.shade_table(
        edges,
        {"portable": one_building(10), "rich": one_building(20)},
        [sh.bin_of(np.array([50.0]), np.array([180.0]), CFG)[0]],
        CFG,
    )
    assert set(table["segment_id"]) == {"s0"}
    assert table["shade_rich"].iloc[0] >= table["shade_portable"].iloc[0]
    assert str(table["shade_portable"].dtype) == "float32"
