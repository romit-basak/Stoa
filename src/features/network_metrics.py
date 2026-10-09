"""Network-structure metrics per segment, within walking radii.

Two graphs:

- **Primal (metric).** Nodes are junctions, edges are segments weighted by length. Per radius r:
  shortest-path betweenness of each segment (pairs whose shortest path is <= r and runs along
  it), and closeness, harmonic closeness, reach and junction count around each node (averaged
  over the segment's two ends).
- **Dual (angular, Space Syntax segment analysis).** Each segment is a node; moving from one
  segment to the next costs the turn angle / 90 degrees, so a right-angle turn costs 1 and going
  straight costs 0. The radius is metric: the distance along the least-angle route, measured
  between segment midpoints (half of each segment's length). Per radius: angular choice
  (betweenness on least-angle routes), node count, total angular depth, and the normalised
  measures NAIN = NC^1.2 / TD and NACH = log(choice + 1) / log(TD + 3) (Hillier, Yang and
  Turner 2012).

The search state on the dual graph is (segment, direction of travel), so a route can't leave a
segment by the same end it entered. Pairs are counted once (s->t and t->s are halved).

The loops are compiled with numba and run in parallel over chunks of sources; each chunk keeps
its own accumulators, so results don't depend on the number of threads. Everything here is
BSD-licensed code (numba, numpy, scipy); cityseer would do the same job but is AGPL-3.0.
"""

from __future__ import annotations

import heapq
import math
import os
import time
from typing import Any

import geopandas as gpd
import numba
import numpy as np
import pandas as pd
from numba import njit, prange

from src.features.common import log

_TOL = 1e-9


@njit(cache=True, inline="always")
def _tight(d_from, cost, d_to):
    """True when d_from + cost equals d_to (the edge lies on a shortest path), within tolerance."""
    return abs(d_from + cost - d_to) <= _TOL * max(1.0, d_to)


# --------------------------------------------------------------------------------------------
# Graph construction (plain numpy / python)
# --------------------------------------------------------------------------------------------


def primal_csr(edges: pd.DataFrame, node_ids: np.ndarray):
    """Undirected CSR over nodes: indptr, neighbour index, edge index, weight (length)."""
    idx = pd.Series(np.arange(len(node_ids)), index=node_ids)
    u = idx[edges["u"]].to_numpy()
    v = idx[edges["v"]].to_numpy()
    w = np.maximum(edges["length_m"].to_numpy(dtype=np.float64), 1e-3)
    e = np.arange(len(edges))
    keep = u != v  # self-loops never lie on a shortest path
    src = np.concatenate([u[keep], v[keep]])
    dst = np.concatenate([v[keep], u[keep]])
    eid = np.concatenate([e[keep], e[keep]])
    wt = np.concatenate([w[keep], w[keep]])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(len(node_ids) + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    return (
        indptr,
        dst[order].astype(np.int64),
        eid[order].astype(np.int64),
        wt[order],
        u,
        v,
    )


def _heading(p: tuple, q: tuple) -> tuple[float, float]:
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _turn_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    c = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    return math.degrees(math.acos(c))


def dual_csr(edges: gpd.GeoDataFrame):
    """Directed state graph for angular analysis.

    State 2*i means travelling along segment i from u to v (leaving at v); 2*i+1 means v to u
    (leaving at u). Returns forward CSR (indptr, target state, angular cost, metric cost) and the
    reverse CSR with the same costs, used to find predecessors.
    """
    n_seg = len(edges)
    coords = [list(g.coords) for g in edges.geometry]
    lengths = np.maximum(edges["length_m"].to_numpy(dtype=np.float64), 1e-3)
    # Heading when arriving at each end, and when departing from each end.
    arrive_v = [_heading(c[-2], c[-1]) for c in coords]  # state 2i arrives at v
    arrive_u = [_heading(c[1], c[0]) for c in coords]  # state 2i+1 arrives at u
    depart_u = [_heading(c[0], c[1]) for c in coords]  # entering at u -> state 2i
    depart_v = [_heading(c[-1], c[-2]) for c in coords]  # entering at v -> state 2i+1

    incident: dict[int, list[tuple[int, int]]] = {}
    for i, (u, v) in enumerate(zip(edges["u"], edges["v"])):
        incident.setdefault(u, []).append((i, 0))
        incident.setdefault(v, []).append((i, 1))

    src, dst, ang, met = [], [], [], []
    for inc in incident.values():
        for a, end_a in inc:
            # Leaving a at this node: if the node is a's v end we were in state 2a, else 2a+1.
            s_from = 2 * a if end_a == 1 else 2 * a + 1
            h_in = arrive_v[a] if end_a == 1 else arrive_u[a]
            for b, end_b in inc:
                if b == a:
                    continue
                s_to = 2 * b if end_b == 0 else 2 * b + 1
                h_out = depart_u[b] if end_b == 0 else depart_v[b]
                src.append(s_from)
                dst.append(s_to)
                ang.append(_turn_deg(h_in, h_out) / 90.0)
                met.append(0.5 * (lengths[a] + lengths[b]))
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    ang = np.asarray(ang, dtype=np.float64)
    met = np.asarray(met, dtype=np.float64)
    n_state = 2 * n_seg

    def csr(a, b):
        order = np.argsort(a, kind="stable")
        ptr = np.zeros(n_state + 1, dtype=np.int64)
        np.add.at(ptr, a + 1, 1)
        return np.cumsum(ptr), b[order], ang[order], met[order]

    fwd = csr(src, dst)
    rev = csr(dst, src)
    return fwd, rev


# --------------------------------------------------------------------------------------------
# Compiled kernels
# --------------------------------------------------------------------------------------------


@njit(cache=True)
def _primal_chunk(sources, indptr, nbr, eid, wt, degree, nw, n_edges, radius):
    n = len(indptr) - 1
    dist = np.full(n, np.inf)
    sigma = np.zeros(n)
    delta = np.zeros(n)
    delta_w = np.zeros(n)
    rank = np.full(n, -1, dtype=np.int64)
    order = np.empty(n, dtype=np.int64)
    btw = np.zeros(n_edges)
    btw_w = np.zeros(n_edges)
    reach = np.zeros(n)
    reach_len = np.zeros(n)
    sumd = np.zeros(n)
    harm = np.zeros(n)
    junc = np.zeros(n)
    for s in sources:
        heap = [(0.0, s)]
        dist[s] = 0.0
        sigma[s] = 1.0
        cnt = 0
        while heap:
            d, x = heapq.heappop(heap)
            if rank[x] >= 0 or d > dist[x]:
                continue
            rank[x] = cnt
            order[cnt] = x
            cnt += 1
            for k in range(indptr[x], indptr[x + 1]):
                y = nbr[k]
                nd = d + wt[k]
                if nd <= radius and nd < dist[y] - _TOL * max(1.0, nd):
                    dist[y] = nd
                    heapq.heappush(heap, (nd, y))
        # Path counts in settled order; a predecessor must have settled earlier.
        for i in range(1, cnt):
            y = order[i]
            for k in range(indptr[y], indptr[y + 1]):
                p = nbr[k]
                if rank[p] >= 0 and rank[p] < i and _tight(dist[p], wt[k], dist[y]):
                    sigma[y] += sigma[p]
        # Closeness-style sums around the source.
        reach[s] = cnt - 1
        for i in range(cnt):
            reach_len[s] += nw[order[i]]
        for i in range(1, cnt):
            y = order[i]
            sumd[s] += dist[y]
            harm[s] += 1.0 / dist[y]
            if degree[y] >= 3:
                junc[s] += 1.0
        # Brandes accumulation onto edges.
        for i in range(cnt - 1, 0, -1):
            y = order[i]
            coeff = (1.0 + delta[y]) / sigma[y]
            coeff_w = (nw[y] + delta_w[y]) / sigma[y]
            for k in range(indptr[y], indptr[y + 1]):
                p = nbr[k]
                if rank[p] >= 0 and rank[p] < i and _tight(dist[p], wt[k], dist[y]):
                    c = sigma[p] * coeff
                    btw[eid[k]] += c
                    delta[p] += c
                    c = sigma[p] * coeff_w
                    btw_w[eid[k]] += nw[s] * c
                    delta_w[p] += c
        for i in range(cnt):
            y = order[i]
            dist[y] = np.inf
            sigma[y] = 0.0
            delta[y] = 0.0
            delta_w[y] = 0.0
            rank[y] = -1
    return btw, btw_w, reach, reach_len, sumd, harm, junc


@njit(parallel=True, cache=True)
def _primal_all(nc, indptr, nbr, eid, wt, degree, nw, n_edges, radius):
    n = len(indptr) - 1
    btw = np.zeros((nc, n_edges))
    btw_w = np.zeros((nc, n_edges))
    reach = np.zeros((nc, n))
    reach_len = np.zeros((nc, n))
    sumd = np.zeros((nc, n))
    harm = np.zeros((nc, n))
    junc = np.zeros((nc, n))
    for c in prange(nc):
        b, bw, r, rl, sd, h, j = _primal_chunk(
            np.arange(c, n, nc), indptr, nbr, eid, wt, degree, nw, n_edges, radius
        )
        btw[c] = b
        btw_w[c] = bw
        reach[c] = r
        reach_len[c] = rl
        sumd[c] = sd
        harm[c] = h
        junc[c] = j
    return (
        btw.sum(axis=0),
        btw_w.sum(axis=0),
        reach.sum(axis=0),
        reach_len.sum(axis=0),
        sumd.sum(axis=0),
        harm.sum(axis=0),
        junc.sum(axis=0),
    )


@njit(cache=True)
def _dual_chunk(sources, fptr, fdst, fang, fmet, rptr, rsrc, rang, rmet, sw, radius):
    n_state = len(fptr) - 1
    n_seg = n_state // 2
    dist = np.full(n_state, np.inf)
    metric = np.full(n_state, np.inf)
    sigma = np.zeros(n_state)
    delta = np.zeros(n_state)
    delta_w = np.zeros(n_state)
    rank = np.full(n_state, -1, dtype=np.int64)
    order = np.empty(n_state, dtype=np.int64)
    # Least angular depth to each segment from this source.
    best = np.full(n_seg, np.inf)
    tsig = np.zeros(n_seg)  # paths to the segment's optimal state(s)
    seen = np.empty(n_seg, dtype=np.int64)
    choice = np.zeros(n_seg)
    choice_w = np.zeros(n_seg)
    reach_len = np.zeros(n_seg)
    node_count = np.zeros(n_seg)
    total_depth = np.zeros(n_seg)
    for s in sources:
        heap = [(0.0, 2 * s), (0.0, 2 * s + 1)]
        for x0 in (2 * s, 2 * s + 1):
            dist[x0] = 0.0
            metric[x0] = 0.0
            sigma[x0] = 1.0
        cnt = 0
        while heap:
            d, x = heapq.heappop(heap)
            if rank[x] >= 0 or d > dist[x]:
                continue
            rank[x] = cnt
            order[cnt] = x
            cnt += 1
            for k in range(fptr[x], fptr[x + 1]):
                y = fdst[k]
                if y // 2 == s:
                    continue
                nm = metric[x] + fmet[k]
                if nm > radius:
                    continue
                nd = d + fang[k]
                if nd < dist[y] - _TOL * max(1.0, nd):
                    dist[y] = nd
                    metric[y] = nm
                    heapq.heappush(heap, (nd, y))
                elif abs(nd - dist[y]) <= _TOL * max(1.0, nd) and nm < metric[y]:
                    metric[y] = nm
        # Path counts; the two source states are the roots (rank 0 and 1, or just one).
        n_seen = 0
        for i in range(cnt):
            y = order[i]
            if y // 2 != s:
                for k in range(rptr[y], rptr[y + 1]):
                    p = rsrc[k]
                    if (
                        rank[p] >= 0
                        and rank[p] < i
                        and metric[p] + rmet[k] <= radius
                        and _tight(dist[p], rang[k], dist[y])
                    ):
                        sigma[y] += sigma[p]
            g = y // 2
            if best[g] == np.inf:
                seen[n_seen] = g
                n_seen += 1
            if dist[y] < best[g] - _TOL * max(1.0, dist[y]):
                best[g] = dist[y]
                tsig[g] = sigma[y]
            elif abs(dist[y] - best[g]) <= _TOL * max(1.0, dist[y]):
                tsig[g] += sigma[y]
        for j in range(n_seen):
            g = seen[j]
            node_count[s] += 1.0
            reach_len[s] += sw[g]
            total_depth[s] += best[g]
        # Accumulate dependencies; a state is a target end only if it is optimal for its segment.
        for i in range(cnt - 1, -1, -1):
            y = order[i]
            g = y // 2
            if g == s:
                continue
            tw = 0.0
            if abs(dist[y] - best[g]) <= _TOL * max(1.0, dist[y]):
                tw = sigma[y] / tsig[g]
            coeff = (tw + delta[y]) / sigma[y]
            coeff_w = (tw * sw[g] + delta_w[y]) / sigma[y]
            for k in range(rptr[y], rptr[y + 1]):
                p = rsrc[k]
                if (
                    rank[p] >= 0
                    and rank[p] < i
                    and metric[p] + rmet[k] <= radius
                    and _tight(dist[p], rang[k], dist[y])
                ):
                    delta[p] += sigma[p] * coeff
                    delta_w[p] += sigma[p] * coeff_w
        for i in range(cnt):
            y = order[i]
            if y // 2 != s:
                choice[y // 2] += delta[y]
                choice_w[y // 2] += sw[s] * delta_w[y]
            dist[y] = np.inf
            metric[y] = np.inf
            sigma[y] = 0.0
            delta[y] = 0.0
            delta_w[y] = 0.0
            rank[y] = -1
        for j in range(n_seen):
            best[seen[j]] = np.inf
            tsig[seen[j]] = 0.0
    return choice, choice_w, reach_len, node_count, total_depth


@njit(parallel=True, cache=True)
def _dual_all(nc, fptr, fdst, fang, fmet, rptr, rsrc, rang, rmet, sw, radius):
    n_seg = (len(fptr) - 1) // 2
    choice = np.zeros((nc, n_seg))
    choice_w = np.zeros((nc, n_seg))
    rlen = np.zeros((nc, n_seg))
    ncount = np.zeros((nc, n_seg))
    tdepth = np.zeros((nc, n_seg))
    for c in prange(nc):
        ch, cw, rl, nn, td = _dual_chunk(
            np.arange(c, n_seg, nc),
            fptr,
            fdst,
            fang,
            fmet,
            rptr,
            rsrc,
            rang,
            rmet,
            sw,
            radius,
        )
        choice[c] = ch
        choice_w[c] = cw
        rlen[c] = rl
        ncount[c] = nn
        tdepth[c] = td
    return (
        choice.sum(axis=0),
        choice_w.sum(axis=0),
        rlen.sum(axis=0),
        ncount.sum(axis=0),
        tdepth.sum(axis=0),
    )


# --------------------------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------------------------


def _n_chunks(n: int) -> int:
    # Sources are interleaved across chunks (chunk c takes c, c + nc, ...), which balances dense
    # and sparse areas. Each chunk holds its own accumulators, so keep the count modest.
    return max(1, min(n, 4 * numba.get_num_threads()))


def node_weights(edges: pd.DataFrame, node_ids: np.ndarray) -> np.ndarray:
    """Half the length of every segment touching each node, so the weights sum to network length."""
    idx = pd.Series(np.arange(len(node_ids)), index=node_ids)
    w = np.zeros(len(node_ids))
    half = edges["length_m"].to_numpy(dtype=np.float64) / 2.0
    np.add.at(w, idx[edges["u"]].to_numpy(), half)
    np.add.at(w, idx[edges["v"]].to_numpy(), half)
    return w


def primal_metrics(
    edges: pd.DataFrame, nodes: pd.DataFrame, radius: float, min_reach: int = 10
) -> pd.DataFrame:
    """Metric betweenness per segment and node-level closeness averaged onto segments.

    Closeness is NaN where fewer than min_reach nodes are within the radius (tiny islands,
    where the ratio is meaningless).
    """
    node_ids = nodes["node_id"].to_numpy()
    indptr, nbr, eid, wt, u, v = primal_csr(edges, node_ids)
    degree = nodes["degree"].to_numpy(dtype=np.int64)
    btw, btw_w, reach, reach_len, sumd, harm, junc = _primal_all(
        _n_chunks(len(node_ids)),
        indptr,
        nbr,
        eid,
        wt,
        degree,
        node_weights(edges, node_ids),
        len(edges),
        float(radius),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        close = np.where((sumd > 0) & (reach >= min_reach), reach / sumd, np.nan)
    r = int(radius)
    # Self-loop segments (u == v) take their single node's values.
    return pd.DataFrame(
        {
            "segment_id": edges["segment_id"].to_numpy(),
            f"met_betweenness_{r}": btw / 2.0,
            # Pairs weighted by the network length around each end, in km^2, so a densely
            # mapped area (many short paths) doesn't count for more than its length.
            f"met_betweenness_lw_{r}": btw_w / 2.0 / 1e6,
            f"met_closeness_{r}": (close[u] + close[v]) / 2.0,
            f"met_harmonic_{r}": (harm[u] + harm[v]) / 2.0,
            f"met_reach_{r}": (reach[u] + reach[v]) / 2.0,
            f"met_reach_len_{r}": (reach_len[u] + reach_len[v]) / 2.0,
            f"junctions_{r}": (junc[u] + junc[v]) / 2.0,
        }
    )


def angular_metrics(
    edges: gpd.GeoDataFrame, radius: float, dual=None, min_count: int = 10
) -> pd.DataFrame:
    """Angular choice and integration per segment (Space Syntax segment analysis).

    Integration is NaN where fewer than min_count segments are within the radius.
    """
    (fptr, fdst, fang, fmet), (rptr, rsrc, rang, rmet) = dual or dual_csr(edges)
    n_seg = len(edges)
    sw = edges["length_m"].to_numpy(dtype=np.float64)
    choice, choice_w, reach_len, nc, td = _dual_all(
        _n_chunks(n_seg),
        fptr,
        fdst,
        fang,
        fmet,
        rptr,
        rsrc,
        rang,
        rmet,
        sw,
        float(radius),
    )
    choice = choice / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        nain = np.where((td > 0) & (nc >= min_count), nc**1.2 / td, np.nan)
        nach = np.log(choice + 1.0) / np.log(td + 3.0)
    r = int(radius)
    return pd.DataFrame(
        {
            "segment_id": edges["segment_id"].to_numpy(),
            f"ang_choice_{r}": choice,
            f"ang_choice_lw_{r}": choice_w / 2.0 / 1e6,
            f"ang_node_count_{r}": nc,
            f"ang_reach_len_{r}": reach_len,
            f"ang_total_depth_{r}": td,
            f"ang_integration_{r}": nain,
            f"ang_choice_norm_{r}": nach,
        }
    )


def network_metrics(
    edges: gpd.GeoDataFrame, nodes: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """All metrics for every configured radius, plus node degree, and runtimes in seconds."""
    workers = cfg.get("workers") or os.cpu_count()
    numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))
    out = pd.DataFrame({"segment_id": edges["segment_id"].to_numpy()})
    deg = nodes.set_index("node_id")["degree"]
    out["degree_min"] = np.minimum(
        deg[edges["u"]].to_numpy(), deg[edges["v"]].to_numpy()
    )
    out["degree_max"] = np.maximum(
        deg[edges["u"]].to_numpy(), deg[edges["v"]].to_numpy()
    )
    runtimes = {}
    t = time.perf_counter()
    dual = dual_csr(edges)
    runtimes["dual_build"] = time.perf_counter() - t
    for r in cfg["radii_m"]:
        t = time.perf_counter()
        out = out.merge(primal_metrics(edges, nodes, r), on="segment_id")
        runtimes[f"primal_{r}"] = time.perf_counter() - t
        log.info("primal metrics r=%d m: %.0f s", r, runtimes[f"primal_{r}"])
        t = time.perf_counter()
        out = out.merge(angular_metrics(edges, r, dual), on="segment_id")
        runtimes[f"angular_{r}"] = time.perf_counter() - t
        log.info("angular metrics r=%d m: %.0f s", r, runtimes[f"angular_{r}"])
    return out, runtimes
