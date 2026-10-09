"""Build the walking graph from the extracted ways.

Ways are split wherever they share a node with another walkable way (and at their ends), so each
segment runs between two junctions. segment_id is "<way_id>-<k>", where k counts the pieces of
that way from its first node. IDs are stable for one extract and traceable back to OSM, but they
change when the extract changes, so pin the extract (its SHA-256 is in extract_meta.json).

Outputs in data/interim/osm/<site>/graph/: edges.parquet and nodes.parquet.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point

from src.features.common import Site, log, write_parquet

_SIDEWALK_VALUES = {"both", "left", "right", "no", "none", "separate", "yes"}


def split_ways(ways: pd.DataFrame) -> pd.DataFrame:
    """One row per segment: segment_id, way_id, k, u, v, coords (list of (x, y))."""
    uses = Counter()
    for refs in ways["node_ids"]:
        uses.update(refs)
        # Ends always split; a node repeated inside one way (a loop) splits too.
        uses[refs[0]] += 1
        uses[refs[-1]] += 1
    rows = []
    for way_id, refs, geom in zip(ways["way_id"], ways["node_ids"], ways.geometry):
        coords = list(geom.coords)
        start, k = 0, 0
        for i in range(1, len(refs)):
            if uses[refs[i]] > 1 or i == len(refs) - 1:
                rows.append(
                    {
                        "segment_id": f"{way_id}-{k}",
                        "way_id": way_id,
                        "k": k,
                        "u": refs[start],
                        "v": refs[i],
                        "coords": coords[start : i + 1],
                    }
                )
                start, k = i, k + 1
    return pd.DataFrame(rows)


def axial_bearing(coords: list[tuple[float, float]]) -> float:
    """Bearing of the line from first to last point, in degrees from north, folded to 0-180."""
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 180.0


def _width_m(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace("m", "").strip())
    except ValueError:
        return None


def sidewalk_state(tags: dict[str, Any]) -> str | None:
    """Normalise the sidewalk tagging on a road to both/left/right/no/separate (None if untagged)."""
    v = tags.get("sidewalk") or tags.get("sidewalk:both")
    if v in _SIDEWALK_VALUES:
        return {"none": "no", "yes": "both"}.get(v, v)
    left, right = tags.get("sidewalk:left"), tags.get("sidewalk:right")
    if left or right:
        if "separate" in (left, right):
            return "separate"
        has = [s for s, t in (("left", left), ("right", right)) if t == "yes"]
        return "both" if len(has) == 2 else (has[0] if has else "no")
    return None


def build_graph(
    ways: gpd.GeoDataFrame, crs: int, fcfg: dict[str, Any]
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    seg = split_ways(ways)
    tag_cols = [c for c in fcfg["way_tags"] if c in ways.columns]
    seg = seg.merge(ways[["way_id"] + tag_cols], on="way_id", how="left")
    geom = [LineString(c) for c in seg["coords"]]
    edges = gpd.GeoDataFrame(seg.drop(columns="coords"), geometry=geom, crs=crs)
    edges["length_m"] = edges.length
    edges["bearing_deg"] = [axial_bearing(c) for c in seg["coords"]]
    edges["width_m"] = edges["width"].map(_width_m)
    edges["sidewalk_state"] = [
        sidewalk_state(t) for t in edges[tag_cols].to_dict("records")
    ]
    edges["is_steps"] = edges["highway"].eq("steps")
    edges["is_crossing"] = edges["footway"].eq("crossing") | edges["highway"].eq(
        "crossing"
    )
    edges["is_sidewalk"] = edges["footway"].eq("sidewalk")
    edges["is_indoor"] = edges["highway"].eq("corridor") | edges["indoor"].isin(
        ["yes", "corridor"]
    )
    paved = set(fcfg["paved_surfaces"])
    edges["surface_paved"] = edges["surface"].map(
        lambda s: None if s is None else s in paved
    )
    edges["lit_yes"] = edges["lit"].map(
        lambda s: None if s is None else s not in ("no", "disused")
    )

    # Nodes: endpoints of segments, with degree and connected-component size.
    ends = pd.concat(
        [
            pd.DataFrame({"node_id": edges["u"], "pt": [c[0] for c in seg["coords"]]}),
            pd.DataFrame({"node_id": edges["v"], "pt": [c[-1] for c in seg["coords"]]}),
        ]
    )
    nodes = ends.drop_duplicates("node_id").reset_index(drop=True)
    idx = pd.Series(np.arange(len(nodes)), index=nodes["node_id"])
    ui, vi = idx[edges["u"]].to_numpy(), idx[edges["v"]].to_numpy()
    degree = np.bincount(np.concatenate([ui, vi]), minlength=len(nodes))
    adj = coo_matrix((np.ones(len(ui)), (ui, vi)), shape=(len(nodes), len(nodes)))
    _, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    nodes = gpd.GeoDataFrame(
        {
            "node_id": nodes["node_id"],
            "degree": degree,
            "component": labels,
            "component_size": sizes[labels],
        },
        geometry=[Point(p) for p in nodes["pt"]],
        crs=crs,
    )
    edges["component_size"] = nodes["component_size"].to_numpy()[ui]
    edges["in_main_component"] = nodes["component"].to_numpy()[ui] == np.argmax(sizes)
    return edges, nodes


def graph(site: Site, fcfg: dict[str, Any]) -> dict[str, Any]:
    ways = gpd.read_parquet(site.interim / "ways.parquet")
    edges, nodes = build_graph(ways, site.crs, fcfg)
    write_parquet(edges, site.graph_dir / "edges.parquet")
    write_parquet(nodes, site.graph_dir / "nodes.parquet")
    main_share = (
        edges.loc[edges.in_main_component, "length_m"].sum() / edges.length_m.sum()
    )
    log.info(
        "graph: %d segments, %d nodes, main component holds %.1f%% of length",
        len(edges),
        len(nodes),
        100 * main_share,
    )
    return {
        "segments": len(edges),
        "nodes": len(nodes),
        "main_component_length_share": round(float(main_share), 4),
    }
