from pathlib import Path

import pytest

from src.features.graph import build_graph
from src.features.osm_extract import read_walkable_ways
from tests.features.conftest import CRS

CROSSINGS = Path(__file__).parent / "fixtures" / "crossings.osm"
BBOX = (144.9598, -37.8107, 144.9610, -37.8093)


@pytest.fixture
def graph(fcfg):
    ways = read_walkable_ways(CROSSINGS, BBOX, fcfg).to_crs(CRS)
    return build_graph(ways, CRS, fcfg)


def test_a_crossing_is_inferred_across_each_arm_of_an_unmarked_junction(graph):
    edges, _ = graph
    inferred = edges[edges["is_inferred"]]
    # At r2 (no mapped crossing within 30 m): north sidewalk end 212 to each south corner.
    assert sorted(inferred["segment_id"]) == ["xing-212-222", "xing-212-223"]
    assert inferred["is_crossing"].all() and inferred["in_metric_graph"].all()
    assert inferred["length_m"].between(15, 30).all()


def test_no_crossing_is_inferred_beside_a_mapped_one(graph):
    edges, _ = graph
    # r1 and r3 have the mapped crossings X and Y: nothing inferred there.
    ids = set(edges.loc[edges["is_inferred"], "segment_id"])
    assert not {i for i in ids if "211" in i or "213" in i or "221" in i or "224" in i}


def test_separate_sidewalk_road_leaves_the_metric_graph_unless_it_bridges(graph):
    edges, _ = graph
    e = edges.set_index("segment_id")
    # Road R splits at r2 into 300-0 (r1-r2) and 300-1 (r2-r3). The side street hangs off r2,
    # so one piece of R must stay to keep it connected; the other leaves the metric graph.
    assert e.loc[["300-0", "300-1"], "in_metric_graph"].sum() == 1
    assert e.loc["301-0", "in_metric_graph"]
    assert e.loc[~e.index.str.startswith("300"), "in_metric_graph"].all()


def test_leaving_roads_out_keeps_the_network_in_one_piece(graph):
    edges, _ = graph
    assert edges["in_main_component"].all()
    from src.features.graph import _components, node_table

    nodes, ui, vi = node_table(edges)
    keep = edges["in_metric_graph"].to_numpy()
    labels = _components(ui[keep], vi[keep], len(nodes))
    used = set(ui[keep]) | set(vi[keep])
    assert len({labels[i] for i in used}) == 1


def test_switching_the_rule_off_keeps_every_edge(fcfg):
    cfg = {**fcfg, "graph": {**fcfg["graph"], "exclude_separate_roads": False}}
    ways = read_walkable_ways(CROSSINGS, BBOX, cfg).to_crs(CRS)
    edges, _ = build_graph(ways, CRS, cfg)
    assert edges["in_metric_graph"].all() and not edges["is_inferred"].any()
