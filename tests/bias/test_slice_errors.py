from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.bias import slice_errors as se

RNG = np.random.default_rng(0)


def test_sensor_features_drop_ids_names_and_split_geometry():
    osm = pd.DataFrame(
        {
            "location_id": [1, 2],
            "segment_id": ["10-0", "11-0"],
            "way_id": [10, 11],
            "name": ["Swanston St", "Lygon St"],
            "bearing_deg": [10.0, 90.0],
            "length_m": [40.0, 80.0],
            "met_betweenness_400": [5.0, 1.0],
            "lit": [True, False],
        }
    )
    out = se.sensor_features(osm)
    assert list(out.columns) == ["met_betweenness_400", "lit"]
    assert out.index.tolist() == [1, 2]
    assert out.dtypes.eq("float32").all()


def test_design_matrix_caps_new_station_distance_before_the_opening():
    hourly = pd.DataFrame({c: 0.0 for c in se.ROW_FEATURES}, index=range(2))
    hourly["post_metro_tunnel"] = [False, True]
    hourly["location_id"] = [7, 7]
    osm = pd.DataFrame(
        {"recent_station_dist_m": [150.0], "tram_stop_dist_m": [40.0]},
        index=pd.Index([7], name="location_id"),
    )
    x = se.design_matrix(hourly, np.array([0, 1]), osm, station_cap_m=2000)
    names = se.feature_names(osm)
    col = names.index("recent_station_dist_m")
    assert x.shape == (2, len(names))
    assert x[:, col].tolist() == [2000, 150]
    assert x[:, names.index("tram_stop_dist_m")].tolist() == [40, 40]


def sample_rows():
    return pd.DataFrame(
        {
            "location_id": [1, 1, 1, 2, 3, 3],
            "precinct": ["CBD", "CBD", "CBD", "CBD", "Carlton", "Carlton"],
        }
    )


def test_unweighted_mitigation_returns_no_weights():
    assert se.weights_for({"name": "none"}, sample_rows(), {}) is None


def test_sensor_equal_gives_every_sensor_the_same_total():
    rows = sample_rows()
    w = se.weights_for({"weight": "sensor_equal"}, rows, {})
    totals = pd.Series(w).groupby(rows["location_id"]).sum()
    assert totals.nunique() == 1
    assert w.mean() == pytest.approx(1)


def test_reference_weights_match_the_reference_shares():
    rows = sample_rows()
    ref = {"precinct": pd.Series({"CBD": 1.0, "Carlton": 3.0})}
    w = se.weights_for({"weight": "reference", "dimension": "precinct"}, rows, ref)
    share = pd.Series(w).groupby(rows["precinct"]).sum() / w.sum()
    assert share["CBD"] == pytest.approx(0.25)
    assert share["Carlton"] == pytest.approx(0.75)


def test_unknown_weight_is_an_error():
    with pytest.raises(ValueError, match="unknown weight"):
        se.weights_for({"weight": "magic"}, sample_rows(), {})


def test_folds_never_train_on_a_held_out_sensor():
    groups = np.repeat(np.arange(20), 50)
    plan = se.fold_plan(groups, n_folds=5, train_rows=300, seed=1)
    assert len(plan) == 5
    held_out = np.concatenate([test for _, test in plan])
    assert np.sort(held_out).tolist() == list(range(len(groups)))
    for train, test in plan:
        assert len(train) == 300
        assert not set(groups[train]) & set(groups[test])
    # The same seed gives the same plan, so every mitigation trains on the same rows.
    again = se.fold_plan(groups, n_folds=5, train_rows=300, seed=1)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(plan, again))


def test_cell_metrics_equal_the_row_level_metrics():
    n = 2000
    groups = pd.Series(RNG.choice(["a", "b", "c"], n))
    sensors = pd.Series(RNG.integers(0, 15, n))
    y = RNG.poisson(50, n).astype(float) + 1
    pred = y * RNG.uniform(0.5, 1.5, n)
    by_group = se.cell_metrics(se.cell_sums(groups, sensors, y, pred)).by_group
    for g in "abc":
        m = (groups == g).to_numpy()
        assert by_group.loc[g, "wape"] == pytest.approx(se.wape(y[m], pred[m]))
        assert by_group.loc[g, "bias"] == pytest.approx(se.bias_share(y[m], pred[m]))
        assert by_group.loc[g, "mae"] == pytest.approx(se.mae(y[m], pred[m]))


def test_bootstrap_interval_is_a_point_for_a_single_sensor_group():
    cells = pd.DataFrame(
        {
            "group": ["solo", "many", "many", "many"],
            "sensor": [1, 2, 3, 4],
            "y": [100.0, 100.0, 100.0, 100.0],
            "abs_err": [30.0, 10.0, 50.0, 90.0],
        }
    )
    ci = se.sensor_bootstrap(cells, reps=500, seed=0)
    assert ci.loc["solo"].tolist() == pytest.approx([0.3, 0.3])
    lo, hi = ci.loc["many"]
    assert lo < 0.5 < hi


def test_slice_errors_flags_a_group_whose_interval_is_above_overall():
    sensors = np.repeat(np.arange(12), 100)
    group = np.where(sensors < 6, "night", "day")
    y = np.full(len(sensors), 100.0)
    # Night sensors are all ~60% off, day sensors ~10%: night's whole interval is above
    # the overall error.
    off = np.where(group == "night", 0.6, 0.1) + RNG.normal(0, 0.01, len(y))
    pred = y * (1 + off)
    rows = pd.DataFrame({"time_of_day": group, "location_id": sensors})
    cfg = {"errors": {"bootstrap_reps": 200, "seed": 0, "worse_margin": 0}}
    out = se.slice_errors(rows, ["time_of_day"], y, pred, cfg).set_index("group")
    assert out.loc["night", "worse"]
    assert not out.loc["day", "worse"]
    assert out.loc["night", "n_sensors"] == 6
    assert out.loc["night", "disparity_ratio"] == pytest.approx(
        out.loc["day", "wape"] / out.loc["night", "wape"]
    )


def test_oof_predictions_cover_every_row():
    n = 600
    x = RNG.normal(size=(n, 3)).astype("float32")
    y = np.exp(1 + x[:, 0]).round()
    groups = np.repeat(np.arange(12), 50)
    plan = se.fold_plan(groups, n_folds=3, train_rows=10_000, seed=0)
    pred = se.oof_predict(
        lambda pos: x[pos], y, plan, lambda pos: None, {"max_iter": 20}
    )
    assert not np.isnan(pred).any()
    assert se.wape(y, pred) < 0.5


def test_is_recent_needs_the_same_settings_and_a_young_run(tmp_path):
    meta = tmp_path / "meta.json"
    now = datetime(2026, 10, 22, tzinfo=UTC)
    assert not se.is_recent(meta, "abc", 7, now)
    meta.write_text(
        f'{{"config_hash": "abc", "run_at": "{(now - timedelta(days=2)).isoformat()}"}}'
    )
    assert se.is_recent(meta, "abc", 7, now)
    assert not se.is_recent(meta, "changed", 7, now)
    assert not se.is_recent(meta, "abc", 1, now)


def test_calibration_window_is_each_sensors_first_days():
    ts = pd.Series(
        pd.to_datetime(
            ["2025-01-01", "2025-01-20", "2025-02-15", "2025-03-01", "2025-03-10"],
            utc=True,
        )
    )
    sensors = np.array([1, 1, 1, 2, 2])
    window = se.calibration_window(sensors, ts, days=28)
    assert window.tolist() == [True, True, False, True, True]


def test_calibration_scales_each_sensor_to_its_window_counts():
    y = np.array([100.0, 100.0, 300.0, 10.0, 10.0])
    pred = np.array([50.0, 50.0, 150.0, 0.0, 0.0])
    sensors = np.array([1, 1, 1, 2, 2])
    window = np.array([True, True, False, True, False])
    out = se.calibrate(pred, y, sensors, window)
    assert out[:3].tolist() == [100.0, 100.0, 300.0]  # sensor 1 scaled by 2
    assert out[3:].tolist() == [0.0, 0.0]  # nothing predicted: left as it was


def test_the_bias_model_is_seeded_so_reruns_repeat():
    # Over 200,000 rows the model subsamples to choose its bins; unseeded, reruns differ.
    from src.bias.slices import load_bias_config

    assert "random_state" in load_bias_config()["errors"]["model"]
