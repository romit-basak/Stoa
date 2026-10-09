"""Per-segment features from the OSM layers: tags, amenities, opening hours, built form, transit.

Distances along the network are measured from the nearer end of a segment. POIs and stops are
snapped to their nearest graph node first (POIs farther than amenities.snap_max_m are dropped
and counted in the run metadata).

Outputs in data/processed/features/<site>/:
- osm_segment_features.parquet: one row per segment (GeoParquet, segment geometry kept for maps)
- osm_segment_open_hours.parquet: (segment_id, hour_of_week) -> open POIs nearby, per category.
  Only segments with at least one categorised POI in range appear; absent rows mean zero.
"""

from __future__ import annotations

import heapq
import json
import time
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.features.common import Site, config_hash, log, write_meta, write_parquet
from src.features.network_metrics import network_metrics, primal_csr
from src.features.opening_hours import parse_opening_hours

HIGHWAY_CLASSES = [
    "footway",
    "path",
    "pedestrian",
    "steps",
    "corridor",
    "living_street",
    "residential",
    "service",
    "unclassified",
    "tertiary",
    "secondary",
    "primary",
    "trunk",
    "cycleway",
    "track",
]


# --------------------------------------------------------------------------------------------
# Tags
# --------------------------------------------------------------------------------------------


def tag_features(edges: pd.DataFrame) -> pd.DataFrame:
    hw = edges["highway"].str.replace("_link", "", regex=False)
    out = pd.DataFrame({"segment_id": edges["segment_id"]})
    for c in HIGHWAY_CLASSES:
        out[f"hw_{c}"] = hw.eq(c)
    out["hw_other"] = ~hw.isin(HIGHWAY_CLASSES)
    for c in ("is_sidewalk", "is_crossing", "is_steps", "is_indoor"):
        out[c] = edges[c].astype(bool)
    out["is_bridge"] = edges["bridge"].notna() & edges["bridge"].ne("no")
    out["is_tunnel"] = edges["tunnel"].notna() & edges["tunnel"].ne("no")
    out["is_covered"] = edges["covered"].isin(["yes", "arcade", "colonnade"])
    sw = edges["sidewalk_state"]
    out["sidewalk_both"] = sw.eq("both")
    out["sidewalk_one_side"] = sw.isin(["left", "right"])
    out["sidewalk_none"] = sw.eq("no")
    out["sidewalk_separate"] = sw.eq("separate")
    out["sidewalk_missing"] = sw.isna()
    # Nullable tags: value as 0/1 or NaN, plus an explicit missing flag.
    for name, col in (("surface_paved", "surface_paved"), ("lit", "lit_yes")):
        out[name] = edges[col].map({True: 1.0, False: 0.0})
        out[f"{name}_missing"] = edges[col].isna()
    out["width_m"] = edges["width_m"]
    out["width_missing"] = edges["width_m"].isna()
    out["length_m"] = edges["length_m"]
    out["bearing_deg"] = edges["bearing_deg"]
    return out


# --------------------------------------------------------------------------------------------
# Network reach from points (amenities)
# --------------------------------------------------------------------------------------------


@njit(cache=True)
def _reach_pairs(sources, indptr, nbr, eid, wt, eu, ev, radius):
    """For each source node: every edge with an end within radius, and that end's distance.

    Returns flat arrays (source index, edge index, distance to nearer end).
    """
    n = len(indptr) - 1
    dist = np.full(n, np.inf)
    stamp = np.full(len(eu), -1, dtype=np.int64)
    touched = np.empty(n, dtype=np.int64)
    cap = 1 << 20
    out_s = np.empty(cap, dtype=np.int64)
    out_e = np.empty(cap, dtype=np.int64)
    out_d = np.empty(cap, dtype=np.float64)
    m = 0
    for si in range(len(sources)):
        s = sources[si]
        heap = [(0.0, s)]
        dist[s] = 0.0
        nt = 0
        touched[nt] = s
        nt += 1
        while heap:
            d, x = heapq.heappop(heap)
            if d > dist[x]:
                continue
            for k in range(indptr[x], indptr[x + 1]):
                y = nbr[k]
                nd = d + wt[k]
                if nd <= radius and nd < dist[y]:
                    if dist[y] == np.inf:
                        touched[nt] = y
                        nt += 1
                    dist[y] = nd
                    heapq.heappush(heap, (nd, y))
        for i in range(nt):
            x = touched[i]
            for k in range(indptr[x], indptr[x + 1]):
                e = eid[k]
                if stamp[e] == si:
                    continue
                stamp[e] = si
                if m == cap:
                    cap *= 2
                    out_s = _grow_i(out_s, cap)
                    out_e = _grow_i(out_e, cap)
                    out_d = _grow_f(out_d, cap)
                out_s[m] = si
                out_e[m] = e
                out_d[m] = min(dist[eu[e]], dist[ev[e]])
                m += 1
        for i in range(nt):
            dist[touched[i]] = np.inf
    return out_s[:m], out_e[:m], out_d[:m]


@njit(cache=True)
def _grow_i(a, cap):
    b = np.empty(cap, dtype=a.dtype)
    b[: len(a)] = a
    return b


@njit(cache=True)
def _grow_f(a, cap):
    b = np.empty(cap, dtype=a.dtype)
    b[: len(a)] = a
    return b


def snap_to_nodes(
    points: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, max_m: float
) -> pd.Series:
    """Index (into nodes) of each point's nearest node within max_m; -1 when none."""
    j = gpd.sjoin_nearest(
        points[["geometry"]].reset_index(drop=True),
        nodes[["geometry"]].reset_index(drop=True),
        max_distance=max_m,
        how="left",
    )
    j = j[~j.index.duplicated(keep="first")]
    return j["index_right"].fillna(-1).astype(int)


def amenity_features(
    edges, nodes, pois, cfg: dict[str, Any], csr
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Counts of POIs per category within each network radius, and the weekly open-hours table."""
    acfg = cfg["amenities"]
    indptr, nbr, eid, wt, eu, ev = csr
    cats = [c for c in pois.columns if c.startswith("cat_")]
    snap = snap_to_nodes(pois, nodes, acfg["snap_max_m"]).to_numpy()
    keep = snap >= 0
    pois = pois[keep].reset_index(drop=True)
    snap = snap[keep]
    # Run one search per distinct node; a POI's weight is its category flags.
    src_nodes, poi_src = np.unique(snap, return_inverse=True)
    rmax = float(max(acfg["radii_m"] + [acfg["open_hours_radius_m"]]))
    t = time.perf_counter()
    ps, pe, pd_ = _reach_pairs(
        src_nodes.astype(np.int64), indptr, nbr, eid, wt, eu, ev, rmax
    )
    log.info(
        "amenity reach: %d (source, segment) pairs in %.0f s",
        len(ps),
        time.perf_counter() - t,
    )

    n_src, n_edges = len(src_nodes), len(edges)
    poi_to_src = csr_matrix(
        (np.ones(len(pois)), (poi_src, np.arange(len(pois)))), (n_src, len(pois))
    )
    cat_mat = pois[cats].to_numpy(dtype=np.float64)
    src_cats = poi_to_src @ cat_mat  # (n_src, n_cat)

    out = pd.DataFrame({"segment_id": edges["segment_id"].to_numpy()})
    for r in acfg["radii_m"]:
        sel = pd_ <= r
        reach = csr_matrix((np.ones(sel.sum()), (pe[sel], ps[sel])), (n_edges, n_src))
        counts = reach @ src_cats
        for i, c in enumerate(cats):
            out[f"poi_{c[4:]}_{r}"] = counts[:, i]

    # Weekly open-hours table for selected categories.
    r = acfg["open_hours_radius_m"]
    oh_cats = [f"cat_{c}" for c in acfg["open_hours_categories"]]
    masks = pois["opening_hours"].map(parse_opening_hours)
    known = masks.notna().to_numpy()
    week = np.zeros((len(pois), 168))
    if known.any():
        week[known] = np.stack(masks[known].to_list()).astype(np.float64)
    sel = pd_ <= r
    reach = csr_matrix((np.ones(sel.sum()), (pe[sel], ps[sel])), (n_edges, n_src))
    frames = []
    seg_ids = edges["segment_id"].to_numpy()
    for c in oh_cats:
        w = pois[c].to_numpy(dtype=np.float64)
        open_src = poi_to_src @ (week * w[:, None])  # (n_src, 168)
        unknown_src = poi_to_src @ ((~known) * w)  # (n_src,)
        total_src = poi_to_src @ w
        open_e = (reach @ open_src).astype(np.float32)
        unknown_e = reach @ unknown_src
        total_e = reach @ total_src
        rows = np.flatnonzero(total_e > 0)
        frames.append(
            pd.DataFrame(
                {
                    "segment_id": np.repeat(seg_ids[rows], 168),
                    "hour_of_week": np.tile(np.arange(168, dtype=np.int16), len(rows)),
                    f"open_{c[4:]}": open_e[rows].ravel(),
                    f"unknown_hours_{c[4:]}": np.repeat(unknown_e[rows], 168).astype(
                        np.float32
                    ),
                }
            ).set_index(["segment_id", "hour_of_week"])
        )
    open_hours = pd.concat(frames, axis=1).fillna(0.0).reset_index()
    meta = {
        "pois_snapped": int(keep.sum()),
        "pois_dropped_too_far": int((~keep).sum()),
        "opening_hours_tagged": int(pois["opening_hours"].notna().sum()),
        "opening_hours_parsed": int(known.sum()),
    }
    return out, open_hours, meta


# --------------------------------------------------------------------------------------------
# Built form, green space, transit
# --------------------------------------------------------------------------------------------


def built_form_features(edges, buildings, green, ped_areas, cfg) -> pd.DataFrame:
    b = cfg["built_form"]["buffer_m"]
    cap = float(cfg["built_form"]["max_distance_m"])
    out = pd.DataFrame({"segment_id": edges["segment_id"].to_numpy()})
    buf = gpd.GeoDataFrame(
        {"i": np.arange(len(edges))}, geometry=edges.buffer(b), crs=edges.crs
    )
    bld = buildings[["geometry"]].reset_index(drop=True)
    bld["geometry"] = bld.geometry.make_valid()
    pairs = gpd.sjoin(buf, bld, predicate="intersects")
    inter = shapely.intersection(
        pairs.geometry.to_numpy(),
        bld.geometry.to_numpy()[pairs["index_right"].to_numpy()],
    )
    area = pd.Series(shapely.area(inter)).groupby(pairs["i"].to_numpy()).sum()
    out["building_coverage"] = (
        (area.reindex(range(len(edges)), fill_value=0) / buf.area).clip(0, 1).to_numpy()
    )
    out["building_count"] = (
        pairs.groupby("i").size().reindex(range(len(edges)), fill_value=0).to_numpy()
    )

    for name, layer in (("green", green), ("ped_area", ped_areas)):
        g = layer[["geometry"]].copy()
        g["geometry"] = g.geometry.make_valid()
        j = gpd.sjoin_nearest(
            edges[["geometry"]].reset_index(drop=True),
            g,
            max_distance=cap,
            distance_col="d",
            how="left",
        )
        d = j["d"].groupby(level=0).min().reindex(range(len(edges)))
        out[f"{name}_dist_m"] = d.fillna(cap).clip(upper=cap).to_numpy()
    out["green_adjacent"] = out["green_dist_m"] <= 20
    return out


def transit_features(edges, nodes, transit, cfg, csr) -> pd.DataFrame:
    """Network distance to the nearest stop of each mode (capped), from the segment's nearer end."""
    indptr, nbr, _, wt, eu, ev = csr
    cap = float(cfg["built_form"]["max_distance_m"])
    n = len(indptr) - 1
    # scipy would sum parallel edges between the same two nodes, so keep the shortest first.
    w = pd.DataFrame({"r": np.repeat(np.arange(n), np.diff(indptr)), "c": nbr, "w": wt})
    w = w.groupby(["r", "c"], sort=False)["w"].min().reset_index()
    g = csr_matrix(
        (w["w"].to_numpy(), (w["r"].to_numpy(), w["c"].to_numpy())), shape=(n, n)
    )
    out = pd.DataFrame({"segment_id": edges["segment_id"].to_numpy()})
    snap = snap_to_nodes(transit, nodes, 100.0).to_numpy()
    groups = {
        "tram_stop": transit["mode"].eq("tram_stop"),
        "bus_stop": transit["mode"].eq("bus_stop"),
        "rail_station": transit["mode"].eq("rail_station") & transit["opened"].isna(),
        "recent_station": transit["mode"].eq("rail_station")
        & transit["opened"].notna(),
    }
    for name, m in groups.items():
        src = np.unique(snap[m.to_numpy() & (snap >= 0)])
        if len(src) == 0:
            out[f"{name}_dist_m"] = cap
            continue
        d = dijkstra(g, directed=False, indices=src, min_only=True, limit=cap)
        out[f"{name}_dist_m"] = np.minimum(np.minimum(d[eu], d[ev]), cap)
    return out


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def segment_features(site: Site, fcfg: dict[str, Any]) -> dict[str, Any]:
    edges = gpd.read_parquet(site.graph_dir / "edges.parquet")
    nodes = gpd.read_parquet(site.graph_dir / "nodes.parquet")
    pois = gpd.read_parquet(site.interim / "pois.parquet")
    buildings = gpd.read_parquet(site.interim / "buildings.parquet")
    green = gpd.read_parquet(site.interim / "green.parquet")
    ped = gpd.read_parquet(site.interim / "ped_areas.parquet")
    transit = gpd.read_parquet(site.interim / "transit.parquet")
    # (indptr, neighbour, edge index, length, u index, v index); u/v cover every edge.
    csr = primal_csr(edges, nodes["node_id"].to_numpy())

    runtimes: dict[str, float] = {}
    t = time.perf_counter()
    feats = tag_features(edges)
    amen, open_hours, amen_meta = amenity_features(edges, nodes, pois, fcfg, csr)
    runtimes["amenities"] = time.perf_counter() - t
    t = time.perf_counter()
    built = built_form_features(edges, buildings, green, ped, fcfg)
    runtimes["built_form"] = time.perf_counter() - t
    t = time.perf_counter()
    trans = transit_features(edges, nodes, transit, fcfg, csr)
    runtimes["transit"] = time.perf_counter() - t
    metrics, metric_rt = network_metrics(edges, nodes, fcfg["network_metrics"])
    runtimes.update(metric_rt)

    for part in (amen, built, trans, metrics):
        feats = feats.merge(part, on="segment_id", how="left", validate="one_to_one")
    feats.insert(1, "way_id", edges["way_id"].to_numpy())
    feats.insert(2, "name", edges["name"].to_numpy())
    feats.insert(3, "highway", edges["highway"].to_numpy())
    feats["in_main_component"] = edges["in_main_component"].to_numpy()
    feats = gpd.GeoDataFrame(feats, geometry=edges.geometry.to_numpy(), crs=edges.crs)

    write_parquet(feats, site.processed / "osm_segment_features.parquet")
    write_parquet(open_hours, site.processed / "osm_segment_open_hours.parquet")
    meta_in = json.loads((site.interim / "extract_meta.json").read_text())
    write_meta(
        site.processed / "osm_segment_features_meta.json",
        site=site.name,
        pbf_sha256=meta_in.get("pbf_sha256"),
        config_hash=config_hash(fcfg),
        segments=len(feats),
        columns=len(feats.columns),
        open_hours_rows=len(open_hours),
        runtimes_s={k: round(v, 1) for k, v in runtimes.items()},
        **amen_meta,
    )
    return {"segments": len(feats), **amen_meta, "runtimes_s": runtimes}
