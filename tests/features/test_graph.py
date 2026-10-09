import pytest

from src.features.graph import axial_bearing, build_graph, sidewalk_state
from src.features.osm_extract import read_walkable_ways
from tests.features.conftest import BBOX, CRS


@pytest.fixture
def graph(osm_file, fcfg):
    ways = read_walkable_ways(osm_file, BBOX, fcfg).to_crs(CRS)
    return build_graph(ways, CRS, fcfg)


def test_ways_split_at_shared_nodes(graph):
    edges, _ = graph
    e = edges.set_index("segment_id")
    # Way 100 (1-2-3) meets way 101 at node 2; way 103 (3-8-10) has no inner junction.
    assert sorted(e.index) == ["100-0", "100-1", "101-0", "101-1", "103-0", "105-0"]
    assert (e.loc["100-0", "u"], e.loc["100-0", "v"]) == (1, 2)
    assert (e.loc["100-1", "u"], e.loc["100-1", "v"]) == (2, 3)
    assert (e.loc["103-0", "u"], e.loc["103-0", "v"]) == (3, 10)


def test_segment_ids_are_deterministic(osm_file, fcfg):
    ways = read_walkable_ways(osm_file, BBOX, fcfg).to_crs(CRS)
    a, _ = build_graph(ways, CRS, fcfg)
    b, _ = build_graph(ways.iloc[::-1].reset_index(drop=True), CRS, fcfg)
    assert sorted(a.segment_id) == sorted(b.segment_id)


def test_nodes_degree_and_components(graph):
    edges, nodes = graph
    n = nodes.set_index("node_id")
    assert n.loc[2, "degree"] == 4
    assert n.loc[1, "degree"] == 1
    assert n.loc[10, "degree"] == 2  # cycleway end meets the residential road
    assert edges["in_main_component"].all()  # everything connects via node 3 and 10


def test_edge_flags(graph):
    edges, _ = graph
    e = edges.set_index("segment_id")
    assert e.loc["100-0", "is_sidewalk"] and e.loc["100-0", "surface_paved"]
    assert e.loc["101-0", "is_crossing"]
    assert e.loc["105-0", "sidewalk_state"] == "separate"
    # 0.0001 degrees of longitude at Melbourne's latitude.
    assert e.loc["100-0", "length_m"] == pytest.approx(8.8, abs=0.5)


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"sidewalk": "both"}, "both"),
        ({"sidewalk": "none"}, "no"),
        ({"sidewalk": "separate"}, "separate"),
        ({"sidewalk:both": "separate"}, "separate"),
        ({"sidewalk:left": "yes", "sidewalk:right": "yes"}, "both"),
        ({"sidewalk:left": "yes", "sidewalk:right": "no"}, "left"),
        ({"sidewalk:right": "separate"}, "separate"),
        ({}, None),
    ],
)
def test_sidewalk_state(tags, expected):
    assert sidewalk_state(tags) == expected


def test_axial_bearing_folds_direction():
    assert axial_bearing([(0, 0), (0, 10)]) == pytest.approx(0.0)
    assert axial_bearing([(0, 10), (0, 0)]) == pytest.approx(0.0)
    assert axial_bearing([(0, 0), (10, 0)]) == pytest.approx(90.0)
    assert axial_bearing([(10, 0), (0, 0)]) == pytest.approx(90.0)
