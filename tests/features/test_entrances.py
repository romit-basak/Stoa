from pathlib import Path

import pytest

from src.features.graph import build_graph, routing_graph
from src.features.osm_extract import (
    _read_layer,
    add_station_entrances,
    build_entrances,
    build_transit,
    read_walkable_ways,
)
from tests.features.conftest import CRS

FIXTURE = Path(__file__).parent / "fixtures" / "entrances.osm"
BBOX = (144.9598, -37.8102, 144.9614, -37.8093)


@pytest.fixture
def points():
    return _read_layer(FIXTURE, "points", BBOX)


@pytest.fixture
def entrances(points, fcfg):
    return build_entrances(points, fcfg["entrances"], CRS)


@pytest.fixture
def routing(fcfg, entrances):
    ways = read_walkable_ways(FIXTURE, BBOX, fcfg).to_crs(CRS)
    edges, _ = build_graph(ways, CRS, fcfg)
    return edges, *routing_graph(edges, entrances, fcfg["entrances"])


def test_pedestrian_entrances_are_kept_and_garages_dropped(entrances):
    e = entrances.set_index("osm_id")
    assert sorted(e.index) == [602, 612, 613, 614, 616]
    assert e.loc[614, "kind"] == "station" and e.loc[612, "kind"] == "building"


def test_each_entrance_joins_the_nearest_node_or_edge(routing):
    _, out, _ = routing
    links = out[out["is_connector"]].set_index("segment_id")
    # 602 is a node of the path and 616 is too far: no link for either.
    assert sorted(links.index) == ["ent-612", "ent-613", "ent-614"]
    assert links.loc["ent-612", "v"] == 601
    assert links.loc["ent-614", "v"] == 603  # past the end of the path: joins its end
    assert links.loc["ent-613", "v"] == -613  # a new node on the path
    assert links["length_m"].max() < 30


def test_a_split_edge_is_replaced_by_pieces_that_add_up(routing):
    edges, out, lineage = routing
    # 700-0 runs 601-602-603; it splits at entrance 602 (its own node) and where 613 joins.
    assert sorted(lineage["segment_id"]) == ["700-0.0", "700-0.1", "700-0.2"]
    assert set(lineage["parent"]) == {"700-0"}
    assert "700-0" not in set(out["segment_id"])
    parent = edges.set_index("segment_id").loc["700-0", "length_m"]
    pieces = out[out["segment_id"].str.startswith("700-0.")]["length_m"].sum()
    assert pieces == pytest.approx(parent)
    cuts = sorted(lineage["to_frac"])[:2]
    assert cuts == [pytest.approx(0.5, abs=0.02), pytest.approx(0.75, abs=0.02)]
    split = out.set_index("segment_id")
    assert split.loc["700-0.0", "v"] == 602 and split.loc["700-0.1", "v"] == -613


def test_the_metric_graph_keeps_its_segment_ids(routing):
    edges, _, _ = routing
    assert sorted(edges["segment_id"]) == ["700-0"]


def test_station_entrances_are_rail_access_points_dated_like_their_station(
    points, fcfg, entrances
):
    recent = [{"name": "Town Hall", "opened": "2025-11-30"}]
    transit = build_transit(points, points.iloc[0:0], fcfg, CRS, recent)
    out = add_station_entrances(transit, entrances, 300)
    rail = out[out["mode"] == "rail_station"].set_index("osm_id")
    assert sorted(rail.index) == [614, 615]
    assert rail.loc[614, "opened"] == "2025-11-30"
