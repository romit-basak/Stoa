from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from src.features.common import load_features_config

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.osm"
BBOX = (144.959, -37.811, 144.961, -37.809)
CRS = 28355


@pytest.fixture(scope="session")
def fcfg():
    return load_features_config()


@pytest.fixture(scope="session")
def osm_file():
    return FIXTURE


def make_network(points: dict[int, tuple[float, float]], links: list[tuple[int, int]]):
    """Edges and nodes frames (projected metres) from node coordinates and (u, v) links."""
    rows = [
        {
            "segment_id": f"s{i}",
            "u": u,
            "v": v,
            "geometry": LineString([points[u], points[v]]),
        }
        for i, (u, v) in enumerate(links)
    ]
    edges = gpd.GeoDataFrame(rows, crs=CRS)
    edges["length_m"] = edges.length
    deg = pd.Series(np.concatenate([edges.u, edges.v])).value_counts()
    nodes = gpd.GeoDataFrame(
        {"node_id": list(points), "degree": [int(deg.get(n, 0)) for n in points]},
        geometry=gpd.points_from_xy(*zip(*points.values())),
        crs=CRS,
    )
    return edges, nodes


@pytest.fixture
def jittered_grid():
    """A 6 x 6 grid with jittered node positions (so shortest paths rarely tie)."""
    rng = np.random.default_rng(0)
    pts = {
        i * 6 + j: (i * 100 + rng.normal(0, 10), j * 100 + rng.normal(0, 10))
        for i in range(6)
        for j in range(6)
    }
    links = []
    for i in range(6):
        for j in range(6):
            if i < 5:
                links.append((i * 6 + j, (i + 1) * 6 + j))
            if j < 5:
                links.append((i * 6 + j, i * 6 + j + 1))
    return make_network(pts, links)
