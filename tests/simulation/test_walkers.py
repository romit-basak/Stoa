import numpy as np
import pandas as pd
import pytest

from src.simulation import walkers

TZ = walkers.TZ


def edges(rows):
    return pd.DataFrame(rows, columns=["u", "v", "segment_id", "length_m"])


def ladder():
    """Origin o, two parallel streets (a and b) to a junction j, then on to d."""
    return edges(
        [
            ("o", "a", "oa", 100.0),
            ("a", "j", "aj", 100.0),
            ("o", "b", "ob", 100.0),
            ("b", "j", "bj", 160.0),
            ("j", "d", "jd", 100.0),
        ]
    )


def test_the_shortest_of_two_parallel_segments_is_kept():
    graph = walkers.build_graph(
        edges([("x", "y", "long", 50.0), ("x", "y", "short", 20.0)])
    )
    assert graph["x"]["y"]["segment_id"] == "short"


def test_routes_reach_each_destination_with_distances_along_them():
    graph = walkers.build_graph(ladder())
    routes = walkers.sample_routes(
        graph, ["o"], {"d": "d"}, draws=30, noise=1.0, seed=1
    )
    assert {r.destination for r in routes} == {"d"}
    assert len(routes) == 2
    for route in routes:
        assert route.segments[-1] == "jd"
        assert route.offsets_m == sorted(route.offsets_m)
        assert route.length_m in (300.0, 360.0)


def test_incidence_marks_the_routes_that_pass_each_sensor():
    routes = [
        walkers.Route("d", ["oa", "aj", "jd"], [50.0, 150.0, 250.0], 300.0),
        walkers.Route("d", ["ob", "bj", "jd"], [50.0, 180.0, 310.0], 360.0),
    ]
    passes, offset = walkers.incidence(routes, {1: "aj", 2: "jd"})
    assert passes.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert offset[1].tolist() == [250.0, 310.0]
    assert np.isnan(offset[0, 1])


def test_the_prior_shares_the_crowd_and_favours_shorter_routes():
    routes = [
        walkers.Route("near", ["x"], [1.0], 100.0),
        walkers.Route("near", ["y"], [1.0], 200.0),
        walkers.Route("far", ["z"], [1.0], 500.0),
    ]
    prior = walkers.route_prior(routes, {"near": 3, "far": 1}, crowd=400, beta=2.0)
    assert prior.sum() == pytest.approx(400)
    assert prior[0] + prior[1] == pytest.approx(300)
    assert prior[0] > prior[1]


def test_fitted_trips_reproduce_counts_that_routes_can_explain():
    passes = np.array([[1.0, 0.0], [1.0, 1.0]])
    targets = np.array([80.0, 200.0])
    trips = walkers.fit_trips(passes, targets, np.zeros(2), prior_weight=1e-6)
    assert (trips >= 0).all()
    assert passes @ trips == pytest.approx(targets, rel=1e-3)


def test_block_shares_sum_to_one_and_fall_back_to_even():
    profile = pd.DataFrame({"block_start": [0, 5, 10], "share": [1.0, 3.0, 0.0]})
    shares = walkers.block_shares(profile, 5)
    assert len(shares) == 12 and shares.sum() == pytest.approx(1.0)
    assert shares[1] == pytest.approx(0.75)
    empty = walkers.block_shares(profile.iloc[0:0], 5)
    assert empty == pytest.approx(np.full(12, 1 / 12))


def test_walkers_pass_later_sensors_after_earlier_ones():
    passes = np.array([[1.0], [1.0]])
    offset = np.array([[100.0], [490.0]])
    shares = np.zeros(12)
    shares[2] = 1.0
    start = pd.Timestamp("2026-10-27 23:00", tz=TZ)
    rng = np.random.default_rng(0)
    trips = np.array([10.0])
    out = walkers.crossings(trips, passes, offset, start, shares, 5, 1.3, rng)
    assert len(out) == 20
    first = out[out["sensor_index"] == 0]["time"]
    later = out[out["sensor_index"] == 1]["time"]
    assert first.min() >= start + pd.Timedelta(minutes=10)
    assert first.max() < start + pd.Timedelta(minutes=15)
    gap = later.reset_index(drop=True) - first.reset_index(drop=True)
    assert gap.dt.total_seconds().tolist() == pytest.approx([300.0] * 10)


def test_blocks_and_comparison_with_the_observed_series():
    start = pd.Timestamp("2026-10-27 23:00", tz=TZ)
    passed = pd.DataFrame(
        {
            "sensor_index": [0, 0, 0, 1],
            "time": [start, start + pd.Timedelta(minutes=6), start, start],
        }
    )
    simulated = walkers.to_blocks(passed, [161, 136], 5)
    assert simulated.set_index(["location_id", "block"])["total"].to_dict() == {
        (136, start): 1,
        (161, start): 2,
        (161, start + pd.Timedelta(minutes=5)): 1,
    }
    observed = pd.DataFrame(
        {
            "location_id": [161, 161, 136],
            "block": [start, start + pd.Timedelta(minutes=5), start],
            "total": [5, 1, 2],
        }
    )
    result = walkers.compare(simulated, observed, fitted=[161]).set_index("location_id")
    assert result.loc[161, "observed_total"] == 6
    assert result.loc[161, "simulated_total"] == 3
    assert result.loc[161, "used_in_fit"]
    assert not result.loc[136, "used_in_fit"]
    assert result.loc[161, "obs_peak_block"] == start
