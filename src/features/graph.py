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

Entrances (building and station entrances) join a separate routing graph, for walkers and
store siting: each gets a link "ent-<osm_id>" to the nearest node, or to the nearest edge,
which is split there into "<segment_id>.0" and "<segment_id>.1" (segment_lineage.parquet maps
the pieces to the segment they came from). The metric graph and segment IDs are unchanged.

Outputs in data/interim/osm/<site>/graph/: edges.parquet and nodes.parquet, plus
routing_edges.parquet and segment_lineage.parquet.
"""

from __future__ import annotations

import itertools
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
from shapely.ops import substring

from src.features.common import Site, log, write_parquet

# An entrance this close to a path is a node of that path.
ON_PATH_M = 0.5
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
    ways: gpd.GeoDataFrame,
    crs: int,
    fcfg: dict[str, Any],
    plazas: gpd.GeoDataFrame | None = None,
    detected: gpd.GeoDataFrame | None = None,
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
    edges["is_area_edge"] = False
    edges["is_detected"] = False
    gcfg = fcfg.get("graph", {})
    if detected is not None and len(detected):
        found = detected_edges(
            edges, detected.to_crs(crs), gcfg.get("detected_snap_m", 5)
        )
        edges = gpd.GeoDataFrame(
            pd.concat([edges, found], ignore_index=True), geometry="geometry", crs=crs
        )
    if (
        plazas is not None
        and len(plazas)
        and gcfg.get("plaza_edges", {}).get("enabled")
    ):
        across = plaza_edges(edges, plazas.to_crs(crs), gcfg["plaza_edges"])
        if len(across):
            edges = gpd.GeoDataFrame(
                pd.concat([edges, across], ignore_index=True),
                geometry="geometry",
                crs=crs,
            )
    edges = add_metric_graph(edges, gcfg)
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


def detected_edges(
    edges: gpd.GeoDataFrame, detected: gpd.GeoDataFrame, snap_m: float
) -> gpd.GeoDataFrame:
    """Paths found in aerial imagery (the segmentation model's output), as graph edges.

    Each line's ends join the nearest node within snap_m; an end with none nearby gets a new
    node (negative id, so it can't clash with OSM). segment_id is "det-<n>" in input order,
    and `surface_class` is kept if the file has it.
    """
    nodes, _, _ = node_table(edges)
    tree = STRtree(list(nodes.geometry))
    rows = []
    for n, row in enumerate(detected.itertuples()):
        line = row.geometry
        ends = []
        for k, pt in enumerate((Point(line.coords[0]), Point(line.coords[-1]))):
            i = tree.query_nearest(pt, max_distance=snap_m)
            ends.append(
                int(nodes["node_id"].iloc[i[0]]) if len(i) else -(10**12 + 2 * n + k)
            )
        rows.append(
            {
                "segment_id": f"det-{n}",
                "way_id": -1,
                "k": 0,
                "u": ends[0],
                "v": ends[1],
                "highway": "path",
                "surface": getattr(row, "surface_class", None),
                "geometry": line,
            }
        )
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=edges.crs)
    out["length_m"] = out.length
    out["bearing_deg"] = [axial_bearing(list(g.coords)) for g in out.geometry]
    for flag in (
        "is_crossing",
        "is_sidewalk",
        "is_steps",
        "is_indoor",
        "is_inferred",
        "is_area_edge",
    ):
        out[flag] = False
    out["is_detected"] = True
    return out


def plaza_edges(
    edges: gpd.GeoDataFrame, plazas: gpd.GeoDataFrame, cfg: dict[str, Any]
) -> gpd.GeoDataFrame:
    """Straight edges across each plaza between the places paths reach it.

    Access points are graph nodes on the plaza's outline (shared OSM nodes) or within
    access_tolerance_m of it. Each is joined to its max_neighbours nearest others where the
    straight line stays inside the plaza, so a concave plaza isn't crossed outside its shape.
    segment_id is "area-<osm_id>-<u>-<v>" with u < v, so IDs don't depend on input order.
    """
    tol, k = cfg["access_tolerance_m"], cfg["max_neighbours"]
    nodes, _, _ = node_table(edges)
    ids = nodes["node_id"].to_numpy()
    pts = nodes.geometry
    tree = STRtree(list(pts))
    rows = []
    for plaza in plazas.itertuples():
        shape = plaza.geometry
        on_outline = set(plaza.node_ids) & set(ids)
        near = tree.query(shape.boundary, predicate="dwithin", distance=tol)
        access = sorted(on_outline | set(ids[near]))
        if len(access) < 2:
            continue
        where = nodes.set_index("node_id").geometry
        xy = np.array([(where[a].x, where[a].y) for a in access])
        inside = shape.buffer(0.5)
        seen = set()
        for i, a in enumerate(access):
            d = np.hypot(*(xy - xy[i]).T)
            for j in np.argsort(d)[1 : k + 1]:
                b = access[j]
                pair = (min(a, b), max(a, b))
                if pair in seen or d[j] < 0.5:
                    continue
                seen.add(pair)
                line = LineString([xy[i], xy[j]])
                if inside.covers(line):
                    rows.append((plaza.osm_id, plaza.name, pair, line))
    if not rows:
        return edges.iloc[0:0].copy()
    out = gpd.GeoDataFrame(
        {
            "segment_id": [f"area-{o}-{u}-{v}" for o, _, (u, v), _ in rows],
            "way_id": [o for o, _, _, _ in rows],
            "k": 0,
            "u": [u for _, _, (u, _), _ in rows],
            "v": [v for _, _, (_, v), _ in rows],
            "highway": "pedestrian",
            "name": [n for _, n, _, _ in rows],
        },
        geometry=[g for _, _, _, g in rows],
        crs=edges.crs,
    ).drop_duplicates("segment_id")
    out["length_m"] = out.length
    out["bearing_deg"] = [axial_bearing(list(g.coords)) for g in out.geometry]
    for flag in ("is_crossing", "is_sidewalk", "is_steps", "is_indoor", "is_inferred"):
        out[flag] = False
    out["is_area_edge"] = True
    return out.reset_index(drop=True)


def routing_graph(
    edges: gpd.GeoDataFrame, entrances: gpd.GeoDataFrame, cfg: dict[str, Any]
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """The edges plus a link from each entrance to the network; and the lineage of split edges.

    An entrance that is already a graph node needs no link, and one that is a node partway
    along a path splits that path at itself. Otherwise it joins the nearest node within
    node_snap_m, or the nearest edge within max_link_m at the closest point, which splits that
    edge (virtual node id: minus the entrance's OSM id). Entrances farther away stay
    unconnected.
    """
    nodes, _, _ = node_table(edges)
    node_ids = set(nodes["node_id"])
    node_tree = STRtree(list(nodes.geometry))
    edge_tree = STRtree(list(edges.geometry))
    links, cuts = [], {}
    for e in entrances.sort_values("osm_id").itertuples():
        if e.osm_id in node_ids:
            continue
        i = node_tree.query_nearest(e.geometry, max_distance=cfg["node_snap_m"])
        if len(i):
            target = int(nodes["node_id"].iloc[i[0]])
            end = nodes.geometry.iloc[i[0]]
        else:
            j = edge_tree.query_nearest(e.geometry, max_distance=cfg["max_link_m"])
            if not len(j):
                continue
            line = edges.geometry.iloc[j[0]]
            frac = line.project(e.geometry, normalized=True)
            if frac <= 0.0 or frac >= 1.0:
                at_u = frac <= 0.0
                target = int(edges["u"].iloc[j[0]] if at_u else edges["v"].iloc[j[0]])
                end = Point(line.coords[0] if at_u else line.coords[-1])
            elif line.distance(e.geometry) < ON_PATH_M:
                # The entrance is a node of the path itself: split there, no link needed.
                cuts.setdefault(j[0], []).append((frac, int(e.osm_id), e.geometry))
                continue
            else:
                target = -int(e.osm_id)
                end = line.interpolate(frac, normalized=True)
                cuts.setdefault(j[0], []).append((frac, target, end))
        links.append(
            {
                "segment_id": f"ent-{e.osm_id}",
                "u": int(e.osm_id),
                "v": target,
                "entrance_kind": e.kind,
                "geometry": LineString([e.geometry, end]),
            }
        )
    pieces, lineage, split = [], [], sorted(cuts)
    for j in split:
        parent = edges.iloc[j]
        stops = sorted(cuts[j])
        bounds = (
            [(0.0, int(parent["u"]))]
            + [(f, n) for f, n, _ in stops]
            + [(1.0, int(parent["v"]))]
        )
        for k, ((f0, a), (f1, b)) in enumerate(itertools.pairwise(bounds)):
            child = parent.copy()
            child["segment_id"] = f"{parent['segment_id']}.{k}"
            child["u"], child["v"] = a, b
            child["geometry"] = substring(parent.geometry, f0, f1, normalized=True)
            child["length_m"] = child["geometry"].length
            pieces.append(child)
            lineage.append(
                {
                    "segment_id": child["segment_id"],
                    "parent": parent["segment_id"],
                    "from_frac": f0,
                    "to_frac": f1,
                }
            )
    kept = edges.drop(index=edges.index[split])
    parts = [kept]
    if pieces:
        parts.append(gpd.GeoDataFrame(pieces, geometry="geometry", crs=edges.crs))
    if links:
        conn = gpd.GeoDataFrame(links, geometry="geometry", crs=edges.crs)
        conn["length_m"] = conn.length
        conn["highway"] = "footway"
        conn["is_connector"] = True
        parts.append(conn)
    out = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=edges.crs
    )
    out["is_connector"] = out.get("is_connector", False)
    out["is_connector"] = out["is_connector"].fillna(False).astype(bool)
    lineage_df = pd.DataFrame(
        lineage, columns=["segment_id", "parent", "from_frac", "to_frac"]
    )
    return out, lineage_df


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
    edges["is_area_edge"] = edges["is_area_edge"].fillna(False).astype(bool)
    edges["is_detected"] = edges["is_detected"].fillna(False).astype(bool)
    edges["in_metric_graph"] = keep_bridges(edges, ~excluded)
    return edges


def graph(site: Site, fcfg: dict[str, Any]) -> dict[str, Any]:
    ways = gpd.read_parquet(site.interim / "ways.parquet")
    plazas_path = site.interim / "plazas.parquet"
    plazas = gpd.read_parquet(plazas_path) if plazas_path.exists() else None
    # Paths detected in aerial imagery, once a segmentation model writes them (Tier 1 item 5).
    detected_path = site.interim / "detected_paths.parquet"
    detected = gpd.read_parquet(detected_path) if detected_path.exists() else None
    edges, nodes = build_graph(ways, site.crs, fcfg, plazas, detected)
    write_parquet(edges, site.graph_dir / "edges.parquet")
    write_parquet(nodes, site.graph_dir / "nodes.parquet")
    entrances_path = site.interim / "entrances.parquet"
    if entrances_path.exists() and "entrances" in fcfg:
        routing, lineage = routing_graph(
            edges, gpd.read_parquet(entrances_path), fcfg["entrances"]
        )
        write_parquet(routing, site.graph_dir / "routing_edges.parquet")
        write_parquet(lineage, site.graph_dir / "segment_lineage.parquet")
        log.info(
            "routing graph: %d entrance links, %d edges split",
            int(routing["is_connector"].sum()),
            lineage["parent"].nunique() if len(lineage) else 0,
        )
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
