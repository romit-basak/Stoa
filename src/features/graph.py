"""Build the walking graph from the extracted ways.

Ways are split wherever they share a node with another walkable way (and at their ends), so each
segment runs between two junctions. segment_id is "<way_id>-<k>", where k counts the pieces of
that way from its first node. IDs are stable for one extract and traceable back to OSM, but they
change when the extract changes, so pin the extract (its SHA-256 is in extract_meta.json).

Network metrics run on the metric graph, a subset of the edges (`in_metric_graph`). Road
centrelines whose sidewalks are mapped as their own ways (sidewalk=separate) are left out of
it: otherwise betweenness and choice route through the road and its two sidewalks at once.
Leaving them out would cut the sidewalks apart wherever no crossing is mapped, so first a
crossing is inferred at each such junction (segment_id "xing-<u>-<v>", is_inferred), and a
road piece stays in after all if the network would still fall apart without it.

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
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely import STRtree
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

    edges["is_inferred"] = False
    edges = add_metric_graph(edges, fcfg.get("graph", {}))
    edges, nodes = add_components(edges)
    return edges, nodes


def node_table(
    edges: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """Endpoints of the edges with degree and connected-component size; and u, v positions."""
    first = [g.coords[0] for g in edges.geometry]
    last = [g.coords[-1] for g in edges.geometry]
    ends = pd.concat(
        [
            pd.DataFrame({"node_id": edges["u"].to_numpy(), "pt": first}),
            pd.DataFrame({"node_id": edges["v"].to_numpy(), "pt": last}),
        ]
    )
    nodes = ends.drop_duplicates("node_id").reset_index(drop=True)
    idx = pd.Series(np.arange(len(nodes)), index=nodes["node_id"])
    ui, vi = idx[edges["u"]].to_numpy(), idx[edges["v"]].to_numpy()
    degree = np.bincount(np.concatenate([ui, vi]), minlength=len(nodes))
    labels = _components(ui, vi, len(nodes))
    sizes = np.bincount(labels)
    table = gpd.GeoDataFrame(
        {
            "node_id": nodes["node_id"],
            "degree": degree,
            "component": labels,
            "component_size": sizes[labels],
        },
        geometry=[Point(p) for p in nodes["pt"]],
        crs=edges.crs,
    )
    return table, ui, vi


def _components(ui: np.ndarray, vi: np.ndarray, n: int) -> np.ndarray:
    adj = coo_matrix((np.ones(len(ui)), (ui, vi)), shape=(n, n))
    return connected_components(adj, directed=False)[1]


def add_components(
    edges: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes, ui, _ = node_table(edges)
    sizes = np.bincount(nodes["component"].to_numpy())
    edges = edges.copy()
    edges["component_size"] = nodes["component_size"].to_numpy()[ui]
    edges["in_main_component"] = nodes["component"].to_numpy()[ui] == np.argmax(sizes)
    return edges, nodes


# --------------------------------------------------------------------------------------------
# The metric graph: separate-sidewalk roads out, crossings inferred where none is mapped
# --------------------------------------------------------------------------------------------


def separate_roads(edges: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """True for road centrelines whose sidewalks are mapped as their own ways."""
    if not cfg.get("exclude_separate_roads"):
        return np.zeros(len(edges), dtype=bool)
    road = edges["highway"].str.removesuffix("_link").isin(cfg["road_classes"])
    return (road & edges["sidewalk_state"].eq("separate")).to_numpy()


def _side(origin: np.ndarray, direction: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Sign of the cross product: which side of a road each point lies on."""
    d = p - origin
    return np.sign(direction[0] * d[:, 1] - direction[1] * d[:, 0])


def infer_crossings(
    edges: gpd.GeoDataFrame, excluded: np.ndarray, cfg: dict[str, Any]
) -> gpd.GeoDataFrame:
    """Crossings at junctions of excluded roads, between sidewalk ends on opposite sides.

    For each excluded road piece and each junction at its ends (3 or more edges meet), the
    closest pair of sidewalk ends within junction_radius_m, one on each side, whose straight
    link crosses that piece and is no longer than max_length_m. The pair is skipped when a
    mapped crossing lies within mapped_crossing_radius_m of the link, or when the sidewalks are
    already joined by a walk shorter than detour_factor times the link.
    """
    ic = cfg["inferred_crossings"]
    nodes, ui, vi = node_table(edges)
    xy = np.array([(p.x, p.y) for p in nodes.geometry])
    sidewalk = edges["is_sidewalk"].fillna(False).to_numpy(dtype=bool)
    ends = np.unique(np.concatenate([ui[sidewalk], vi[sidewalk]]))
    if len(ends) == 0 or not excluded.any():
        return edges.iloc[0:0].copy()
    tree = STRtree([Point(p) for p in xy[ends]])
    crossing_geoms = edges.geometry[
        edges["is_crossing"].fillna(False).to_numpy(dtype=bool)
    ]
    crossing_tree = STRtree(list(crossing_geoms)) if len(crossing_geoms) else None

    # Walking distances on the graph without the excluded roads.
    keep = ~excluded
    w = np.maximum(edges["length_m"].to_numpy(dtype=float), 1e-3)
    n = len(nodes)
    walk = coo_matrix(
        (
            np.concatenate([w[keep], w[keep]]),
            (
                np.concatenate([ui[keep], vi[keep]]),
                np.concatenate([vi[keep], ui[keep]]),
            ),
        ),
        shape=(n, n),
    ).tocsr()

    pairs: dict[tuple[int, int], tuple[float, Any]] = {}
    degree = nodes["degree"].to_numpy()
    for i in np.flatnonzero(excluded):
        line = edges.geometry.iloc[i]
        coords = np.asarray(line.coords)
        for j_pos, nxt in ((ui[i], coords[1]), (vi[i], coords[-2])):
            if degree[j_pos] < 3:
                continue
            junction = xy[j_pos]
            near = ends[tree.query(Point(junction).buffer(ic["junction_radius_m"]))]
            near = near[np.hypot(*(xy[near] - junction).T) <= ic["junction_radius_m"]]
            if len(near) < 2:
                continue
            side = _side(junction, nxt - junction, xy[near])
            left, right = near[side > 0], near[side < 0]
            best = None
            for a in left:
                for b in right:
                    d = float(np.hypot(*(xy[a] - xy[b])))
                    if d > ic["max_length_m"]:
                        continue
                    link = LineString([xy[a], xy[b]])
                    if not link.intersects(line):
                        continue
                    key = (
                        d,
                        min(nodes["node_id"][a], nodes["node_id"][b]),
                        max(nodes["node_id"][a], nodes["node_id"][b]),
                    )
                    if best is None or key < best[0]:
                        best = (key, a, b, link)
            if best is None:
                continue
            (d, id_a, id_b), a, b, link = best
            if crossing_tree is not None and len(
                crossing_tree.query(
                    link, predicate="dwithin", distance=ic["mapped_crossing_radius_m"]
                )
            ):
                continue
            dist = dijkstra(walk, indices=a, limit=ic["detour_factor"] * d)[b]
            if np.isfinite(dist):
                continue
            pairs[(id_a, id_b)] = (d, link)
    rows = [
        {
            "segment_id": f"xing-{a}-{b}",
            "way_id": -1,
            "k": 0,
            "u": a,
            "v": b,
            "highway": "footway",
            "footway": "crossing",
            "geometry": link,
        }
        for (a, b), (_, link) in sorted(pairs.items())
    ]
    if not rows:
        return edges.iloc[0:0].copy()
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=edges.crs)
    out["length_m"] = out.length
    out["bearing_deg"] = [axial_bearing(list(g.coords)) for g in out.geometry]
    out["is_crossing"] = True
    out["is_sidewalk"] = False
    out["is_steps"] = False
    out["is_indoor"] = False
    out["is_inferred"] = True
    return out


def keep_bridges(edges: pd.DataFrame, in_metric: np.ndarray) -> np.ndarray:
    """Put back excluded edges that join parts of the network nothing else joins."""
    nodes = pd.unique(np.concatenate([edges["u"].to_numpy(), edges["v"].to_numpy()]))
    pos = pd.Series(np.arange(len(nodes)), index=nodes)
    u, v = pos[edges["u"]].to_numpy(), pos[edges["v"]].to_numpy()

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Start each component at its own label, then union along kept bridges.
    parent = np.arange(len(nodes))
    for a, b in zip(u[in_metric], v[in_metric]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    out = in_metric.copy()
    for i in np.argsort(edges["segment_id"].to_numpy()):
        if out[i]:
            continue
        ra, rb = find(u[i]), find(v[i])
        if ra != rb:
            parent[ra] = rb
            out[i] = True
    return out


def add_metric_graph(edges: gpd.GeoDataFrame, cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """Inferred crossings appended, and `in_metric_graph` set for every edge."""
    excluded = separate_roads(edges, cfg)
    crossings = (
        infer_crossings(edges, excluded, cfg) if excluded.any() else edges.iloc[0:0]
    )
    if len(crossings):
        edges = pd.concat([edges, crossings], ignore_index=True)
        edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=crossings.crs)
        excluded = np.concatenate([excluded, np.zeros(len(crossings), dtype=bool)])
    edges["is_inferred"] = edges["is_inferred"].fillna(False).astype(bool)
    edges["in_metric_graph"] = keep_bridges(edges, ~excluded)
    return edges


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
