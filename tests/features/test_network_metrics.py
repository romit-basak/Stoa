import heapq
from itertools import combinations, pairwise

import networkx as nx
import numpy as np
import pytest

from src.features.network_metrics import (
    angular_metrics,
    dual_csr,
    node_weights,
    primal_metrics,
)
from tests.features.conftest import make_network

BIG = 1e9


def line(n, step=10.0):
    pts = {i: (i * step, 0.0) for i in range(n)}
    return make_network(pts, [(i, i + 1) for i in range(n - 1)])


def nx_graph(edges):
    g = nx.Graph()
    for r in edges.itertuples():
        g.add_edge(r.u, r.v, weight=r.length_m, sid=r.segment_id)
    return g


def test_line_betweenness_by_hand():
    edges, nodes = line(4)
    m = primal_metrics(edges, nodes, BIG, min_reach=1).set_index("segment_id")
    # Pairs through each edge of a-b-c-d: ab 3, bc 4, cd 3.
    assert m[f"met_betweenness_{int(BIG)}"].tolist() == [3.0, 4.0, 3.0]
    # Within 15 m only neighbours pair up.
    m = primal_metrics(edges, nodes, 15, min_reach=1).set_index("segment_id")
    assert m["met_betweenness_15"].tolist() == [1.0, 1.0, 1.0]
    assert m["met_reach_15"].tolist() == [1.5, 2.0, 1.5]


def test_betweenness_matches_networkx(jittered_grid):
    edges, nodes = jittered_grid
    m = primal_metrics(edges, nodes, BIG).set_index("segment_id")
    g = nx_graph(edges)
    ref = nx.edge_betweenness_centrality(g, weight="weight", normalized=False)
    for (a, b), v in ref.items():
        assert m.loc[
            g.edges[a, b]["sid"], f"met_betweenness_{int(BIG)}"
        ] == pytest.approx(v)


def test_radius_betweenness_and_weights_match_brute_force(jittered_grid):
    edges, nodes = jittered_grid
    r = 250
    g = nx_graph(edges)
    w = node_weights(edges, nodes["node_id"].to_numpy())
    dist = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
    plain = dict.fromkeys(edges.segment_id, 0.0)
    weighted = dict.fromkeys(edges.segment_id, 0.0)
    for s, t in combinations(g.nodes, 2):
        if dist[s][t] <= r:
            paths = list(nx.all_shortest_paths(g, s, t, weight="weight"))
            for p in paths:
                for a, b in pairwise(p):
                    sid = g.edges[a, b]["sid"]
                    plain[sid] += 1 / len(paths)
                    weighted[sid] += w[s] * w[t] / len(paths)
    m = primal_metrics(edges, nodes, r).set_index("segment_id")
    for sid in edges.segment_id:
        assert m.loc[sid, f"met_betweenness_{r}"] == pytest.approx(plain[sid])
        assert m.loc[sid, f"met_betweenness_lw_{r}"] * 1e6 == pytest.approx(
            weighted[sid]
        )


def test_closeness_masked_for_tiny_neighbourhoods():
    edges, nodes = line(3)
    m = primal_metrics(edges, nodes, BIG)  # default min_reach=10
    assert m.filter(like="met_closeness").isna().all().all()


def test_angular_depth_straight_vs_turn():
    straight = make_network({0: (0, 0), 1: (10, 0), 2: (20, 0)}, [(0, 1), (1, 2)])
    turn = make_network({0: (0, 0), 1: (10, 0), 2: (10, 10)}, [(0, 1), (1, 2)])
    a = angular_metrics(straight[0], BIG, min_count=1)
    b = angular_metrics(turn[0], BIG, min_count=1)
    td = f"ang_total_depth_{int(BIG)}"
    assert a[td].tolist() == pytest.approx([0.0, 0.0])
    assert b[td].tolist() == pytest.approx([1.0, 1.0])  # a right angle costs 1


def test_angular_route_cannot_double_back():
    # A T: the stem can reach either arm with a right-angle turn; the arms reach each other
    # straight through. Going arm -> stem -> other arm would need a U-turn inside the stem.
    edges, _ = make_network(
        {0: (-10, 0), 1: (0, 0), 2: (10, 0), 3: (0, -10)}, [(0, 1), (1, 2), (1, 3)]
    )
    m = angular_metrics(edges, BIG, min_count=1).set_index("segment_id")
    td = f"ang_total_depth_{int(BIG)}"
    assert m.loc["s0", td] == pytest.approx(
        0.0 + 1.0
    )  # s1 straight, s2 one right angle
    assert m.loc["s2", td] == pytest.approx(2.0)
    assert m[f"ang_choice_{int(BIG)}"].tolist() == [0.0, 0.0, 0.0]


def brute_force_angular(edges):
    """Choice and total depth by enumerating every least-angle route on the state graph."""
    (fptr, fdst, fang, _), _ = dual_csr(edges)
    out = [[] for _ in range(len(fptr) - 1)]
    for x in range(len(fptr) - 1):
        for k in range(fptr[x], fptr[x + 1]):
            out[x].append((int(fdst[k]), fang[k]))
    n = len(edges)
    choice, depth = np.zeros(n), np.zeros(n)
    for s in range(n):
        dist = {2 * s: 0.0, 2 * s + 1: 0.0}
        heap, done = [(0.0, 2 * s), (0.0, 2 * s + 1)], set()
        while heap:
            d, x = heapq.heappop(heap)
            if x in done:
                continue
            done.add(x)
            for y, c in out[x]:
                if y // 2 != s and d + c < dist.get(y, np.inf) - 1e-9:
                    dist[y] = d + c
                    heapq.heappush(heap, (d + c, y))
        preds = {y: [] for y in dist}
        for x, dx in dist.items():
            for y, c in out[x]:
                if y in dist and y // 2 != s and abs(dx + c - dist[y]) <= 1e-9:
                    preds[y].append(x)

        def paths(y, s=s, preds=preds):
            if y // 2 == s:
                return [[y]]
            return [q + [y] for p in preds[y] for q in paths(p)]

        for t in range(n):
            if t == s:
                continue
            best = min(dist.get(2 * t, np.inf), dist.get(2 * t + 1, np.inf))
            depth[s] += best
            routes = [
                p
                for st in (2 * t, 2 * t + 1)
                if abs(dist.get(st, np.inf) - best) <= 1e-9
                for p in paths(st)
            ]
            for p in routes:
                for x in p[1:-1]:
                    choice[x // 2] += 1 / len(routes)
    return choice / 2, depth


def test_angular_matches_brute_force(jittered_grid):
    edges, _ = jittered_grid
    m = angular_metrics(edges, BIG)
    choice, depth = brute_force_angular(edges)
    assert m[f"ang_choice_{int(BIG)}"].to_numpy() == pytest.approx(choice)
    assert m[f"ang_total_depth_{int(BIG)}"].to_numpy() == pytest.approx(depth)


def test_angular_radius_limits_reach():
    edges, _ = line(10)  # nine 10 m segments in a row
    m = angular_metrics(edges, 25, min_count=1).set_index("segment_id")
    # Midpoint to midpoint is 10 m per step, so 25 m reaches two segments each way.
    assert m.loc["s4", "ang_node_count_25"] == 5
    assert m.loc["s0", "ang_node_count_25"] == 3
    assert m.loc["s4", "ang_reach_len_25"] == pytest.approx(50.0)


def test_results_do_not_depend_on_thread_count(jittered_grid):
    import numba

    edges, nodes = jittered_grid
    before = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = primal_metrics(edges, nodes, 300)
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        b = primal_metrics(edges, nodes, 300)
    finally:
        numba.set_num_threads(before)
    np.testing.assert_allclose(
        a.drop(columns="segment_id"), b.drop(columns="segment_id")
    )
