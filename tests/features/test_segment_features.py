import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src.features.network_metrics import primal_csr
from src.features.segment_features import (
    amenity_features,
    built_form_features,
    tag_features,
    transit_features,
)
from tests.features.conftest import CRS, make_network


@pytest.fixture
def hairpin():
    """Two parallel 200 m footways 10 m apart, joined only at the west end.

    north: 0 ---- 1      south: 2 ---- 3, with link 0-2. Walking from node 1 to node 3 is
    410 m, though they are 10 m apart in a straight line.
    """
    pts = {0: (0.0, 10.0), 1: (200.0, 10.0), 2: (0.0, 0.0), 3: (200.0, 0.0)}
    edges, nodes = make_network(pts, [(0, 1), (2, 3), (0, 2)])
    return edges, nodes


def pois_at(xy_and_cats, hours=None):
    rows = []
    for i, ((x, y), cat) in enumerate(xy_and_cats):
        rows.append(
            {
                "cat_food_drink": cat == "food_drink",
                "cat_retail": cat == "retail",
                "cat_culture_entertainment": False,
                "cat_services": False,
                "opening_hours": (hours or {}).get(i),
                "geometry": Point(x, y),
            }
        )
    return gpd.GeoDataFrame(rows, crs=CRS)


def small_cfg(fcfg):
    cfg = {**fcfg}
    cfg["amenities"] = {
        **fcfg["amenities"],
        "radii_m": [100, 400],
        "open_hours_radius_m": 100,
        "open_hours_categories": ["food_drink", "retail"],
    }
    return cfg


def test_amenities_count_by_network_not_straight_line(hairpin, fcfg):
    edges, nodes = hairpin
    # A cafe at the east end of the south path: 10 m from node 1 in a straight line.
    pois = pois_at([((200.0, 1.0), "food_drink")])
    out, _, meta = amenity_features(
        edges, nodes, pois, small_cfg(fcfg), primal_csr(edges, nodes.node_id.to_numpy())
    )
    out = out.set_index("segment_id")
    assert out.loc["s1", "poi_food_drink_100"] == 1  # the south path itself
    # The north path's east end (node 1) is 10 m away in a straight line but 410 m on foot;
    # its west end (node 0) is 210 m on foot. So it counts at 400 m, not at 100 m.
    assert out.loc["s0", "poi_food_drink_100"] == 0
    assert out.loc["s0", "poi_food_drink_400"] == 1
    assert out.loc["s2", "poi_food_drink_100"] == 0  # link: node 2 is 200 m away
    assert out.loc["s2", "poi_food_drink_400"] == 1
    assert meta["pois_snapped"] == 1


def test_far_pois_are_dropped(hairpin, fcfg):
    edges, nodes = hairpin
    pois = pois_at([((100.0, 500.0), "retail")])
    out, _, meta = amenity_features(
        edges, nodes, pois, small_cfg(fcfg), primal_csr(edges, nodes.node_id.to_numpy())
    )
    assert meta["pois_dropped_too_far"] == 1
    assert out["poi_retail_100"].sum() == 0


def test_open_hours_table(hairpin, fcfg):
    edges, nodes = hairpin
    pois = pois_at(
        [((0.0, 1.0), "retail"), ((1.0, 1.0), "retail")],
        hours={0: "Mo-Fr 09:00-17:00", 1: "by appointment"},
    )
    _, oh, meta = amenity_features(
        edges, nodes, pois, small_cfg(fcfg), primal_csr(edges, nodes.node_id.to_numpy())
    )
    assert meta["opening_hours_parsed"] == 1
    link = oh[oh.segment_id == "s2"].set_index("hour_of_week")
    assert len(link) == 168
    assert link.loc[10, "open_retail"] == 1  # Monday 10:00
    assert link.loc[5 * 24 + 10, "open_retail"] == 0  # Saturday
    assert (link["unknown_hours_retail"] == 1).all()
    # The north path's far end is 200 m from both shops, but its west end is within 100 m.
    assert set(oh.segment_id) == {"s0", "s1", "s2"}


def test_tag_features_flags_missing():
    edges = pd.DataFrame(
        {
            "segment_id": ["a", "b"],
            "highway": ["footway", "primary_link"],
            "is_sidewalk": [True, False],
            "is_crossing": [False, False],
            "is_steps": [False, False],
            "is_indoor": [False, False],
            "bridge": [None, "yes"],
            "tunnel": [None, None],
            "covered": [None, None],
            "sidewalk_state": [None, "both"],
            "surface_paved": [True, None],
            "lit_yes": [None, False],
            "width_m": [2.0, None],
            "length_m": [10.0, 20.0],
            "bearing_deg": [0.0, 90.0],
        }
    )
    t = tag_features(edges).set_index("segment_id")
    assert t.loc["b", "hw_primary"] and t.loc["b", "is_bridge"]
    assert (
        t.loc["a", "surface_paved"] == 1.0 and not t.loc["a", "surface_paved_missing"]
    )
    assert np.isnan(t.loc["b", "surface_paved"]) and t.loc["b", "surface_paved_missing"]
    assert t.loc["b", "lit"] == 0.0
    assert t.loc["a", "sidewalk_missing"] and t.loc["b", "sidewalk_both"]


def test_built_form(hairpin, fcfg):
    edges, _ = hairpin
    buildings = gpd.GeoDataFrame(geometry=[box(50, 30, 150, 40)], crs=CRS)
    green = gpd.GeoDataFrame(geometry=[box(0, -100, 200, -50)], crs=CRS)
    ped = gpd.GeoDataFrame(geometry=[], crs=CRS)
    out = built_form_features(edges, buildings, green, ped, fcfg).set_index(
        "segment_id"
    )
    assert out.loc["s0", "building_count"] == 1 and out.loc["s1", "building_count"] == 0
    assert 0 < out.loc["s0", "building_coverage"] < 1
    assert out.loc["s1", "green_dist_m"] == pytest.approx(50.0)
    assert out["ped_area_dist_m"].eq(fcfg["built_form"]["max_distance_m"]).all()


def test_transit_distance_is_network_and_capped(hairpin, fcfg):
    edges, nodes = hairpin
    transit = gpd.GeoDataFrame(
        {"mode": ["tram_stop"], "opened": [None]}, geometry=[Point(200.0, 1.0)], crs=CRS
    )
    out = transit_features(
        edges, nodes, transit, fcfg, primal_csr(edges, nodes.node_id.to_numpy())
    )
    out = out.set_index("segment_id")
    assert out.loc["s1", "tram_stop_dist_m"] == 0.0
    assert out.loc["s2", "tram_stop_dist_m"] == pytest.approx(200.0)
    assert out.loc["s0", "tram_stop_dist_m"] == pytest.approx(
        210.0
    )  # via the link at node 0
    cap = fcfg["built_form"]["max_distance_m"]
    assert out["bus_stop_dist_m"].eq(cap).all()
