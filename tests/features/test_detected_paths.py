import geopandas as gpd
from shapely.geometry import LineString

from src.features.graph import build_graph
from src.features.osm_extract import read_walkable_ways
from tests.features.conftest import BBOX, CRS


def test_detected_paths_join_the_graph_as_flagged_edges(osm_file, fcfg):
    ways = read_walkable_ways(osm_file, BBOX, fcfg).to_crs(CRS)
    base, nodes = build_graph(ways, CRS, fcfg)
    n = nodes.set_index("node_id").geometry
    # A worn path from near node 1 (2 m off) to open ground 40 m away.
    start = (n[1].x + 2, n[1].y)
    found = gpd.GeoDataFrame(
        {"surface_class": ["grass"]},
        geometry=[LineString([start, (start[0], start[1] + 40)])],
        crs=CRS,
    )
    edges, _ = build_graph(ways, CRS, fcfg, detected=found)
    det = edges[edges["is_detected"]].iloc[0]
    assert det["segment_id"] == "det-0" and det["surface"] == "grass"
    assert det["u"] == 1  # snapped to the node
    assert det["v"] < 0  # a new node where nothing was mapped
    assert det["in_metric_graph"]
    assert len(edges) == len(base) + 1
