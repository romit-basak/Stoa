from pathlib import Path

import pytest

from src.features.graph import build_graph
from src.features.osm_extract import read_plazas, read_walkable_ways
from tests.features.conftest import CRS

PLAZA = Path(__file__).parent / "fixtures" / "plaza.osm"
BBOX = (144.9597, -37.8102, 144.9605, -37.8095)


@pytest.fixture
def plazas():
    return read_plazas(PLAZA, BBOX)


@pytest.fixture
def graph(fcfg, plazas):
    ways = read_walkable_ways(PLAZA, BBOX, fcfg).to_crs(CRS)
    return build_graph(ways, CRS, fcfg, plazas)


def test_plazas_are_read_with_their_outline_nodes(plazas):
    assert len(plazas) == 1
    p = plazas.iloc[0]
    assert (p["osm_id"], p["osm_type"], p["name"]) == (400, "way", "Test Square")
    assert p["node_ids"] == [501, 502, 503, 504, 505, 506]
    assert p.geometry.geom_type == "Polygon"


def test_the_plaza_area_itself_is_not_a_walkable_way(fcfg):
    ways = read_walkable_ways(PLAZA, BBOX, fcfg)
    assert 400 not in set(ways["way_id"])


def test_edges_join_access_points_inside_the_plaza_only(graph):
    edges, _ = graph
    area = edges[edges["is_area_edge"]]
    # p1-p2 and p1-p5 stay inside the L; p2-p5 would cut the inside corner.
    assert sorted(area["segment_id"]) == ["area-400-501-502", "area-400-501-505"]
    assert (area["highway"] == "pedestrian").all() and area["in_metric_graph"].all()


def test_the_footways_meet_through_the_plaza(graph):
    edges, _ = graph
    assert edges["in_main_component"].all()


def test_without_plazas_the_footways_stay_apart(fcfg):
    ways = read_walkable_ways(PLAZA, BBOX, fcfg).to_crs(CRS)
    edges, _ = build_graph(ways, CRS, fcfg)
    assert not edges["is_area_edge"].any()
    assert not edges["in_main_component"].all()


def test_plaza_ids_do_not_depend_on_input_order(fcfg, plazas):
    ways = read_walkable_ways(PLAZA, BBOX, fcfg).to_crs(CRS)
    a, _ = build_graph(ways, CRS, fcfg, plazas)
    b, _ = build_graph(ways.iloc[::-1].reset_index(drop=True), CRS, fcfg, plazas)
    assert sorted(a.segment_id) == sorted(b.segment_id)
