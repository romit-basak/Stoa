import numpy as np
import pandas as pd
import pytest

from src.features import training_frame as tf
from src.features.common import load_features_config
from src.ingest.common import load_config


def test_only_the_training_site_may_train():
    data_cfg, fcfg = load_config(), load_features_config()
    tf.check_site(fcfg["counts_hourly"]["site"], data_cfg, fcfg)
    for other in ("nyc", "neu_boston"):
        with pytest.raises(ValueError, match="not a training site"):
            tf.check_site(other, data_cfg, fcfg)


def test_rows_need_a_located_outdoor_sensor_and_no_outage():
    hourly = pd.DataFrame(
        {
            "has_location": [True, True, False, True],
            "location_type": ["Outdoor", "Outdoor", None, "Indoor"],
            "suspect_outage_zero": [False, True, False, False],
        }
    )
    assert tf.eligible(hourly).tolist() == [0]


def test_features_never_include_ids_the_target_or_its_parts():
    for variant in tf.VARIANTS:
        names = tf.row_features(variant) + tf.segment_features(variant)
        assert not {"location_id", "segment_id", "way_id", "pedestriancount"} & set(
            names
        )
        assert not {"direction_1", "direction_2", "hour_of_week", "length_m"} & set(
            names
        )
    assert set(tf.row_features("portable")) < set(tf.row_features("rich"))
    assert "post_metro_tunnel" not in tf.row_features("portable")
    with pytest.raises(ValueError, match="unknown variant"):
        tf.row_features("everything")


def toy_frame(n=7):
    hourly = pd.DataFrame({c: np.arange(n, dtype=float) for c in tf.row_features()})
    hourly["location_id"] = [1, 1, 2, 2, 2, 1, 2][:n]
    hourly["post_metro_tunnel"] = [False] * 3 + [True] * (n - 3)
    hourly["pedestriancount"] = np.arange(n)
    osm = pd.DataFrame(
        {"recent_station_dist_m": [120.0, 300.0], "lit": [1.0, 0.0]},
        index=pd.Index([1, 2], name="location_id"),
    )
    return tf.Frame(hourly, osm, tf.row_features(), station_cap_m=2000)


def test_new_stations_are_capped_before_they_opened():
    f = toy_frame()
    X = f.X()
    col = f.columns.index("recent_station_dist_m")
    assert X[:3, col].tolist() == [2000, 2000, 2000]  # before 2025-11-30
    assert X[3:, col].tolist() == [300, 300, 120, 300]  # their real distances after


def test_chunked_matrix_equals_the_whole_one():
    f = toy_frame()
    assert np.array_equal(f.X(chunk=2), f.X(chunk=100))
    assert np.array_equal(f.X(np.array([4, 0])), f.X()[[4, 0]])


def sensors_at(*points):
    return pd.DataFrame([{"location_id": lid, "x": x, "y": y} for lid, x, y in points])


def test_close_neighbours_share_a_group_and_a_fold():
    s = sensors_at((1, 0, 0), (2, 150, 0), (4, 5000, 0), (5, 9000, 0), (6, 13000, 0))
    out = tf.splits(s, {}, n_folds=3, seed=0, radius_m=200).set_index("location_id")
    assert out.loc[1, "group"] == out.loc[2, "group"]
    assert out.loc[1, "fold"] == out.loc[2, "fold"]
    assert out["fold"].between(0, 2).all()


def test_a_chain_of_sensors_does_not_become_one_group():
    # Sensors every 150 m along a street: neighbours pair up, but no group spans the street.
    s = sensors_at(*[(i, 150 * i, 0) for i in range(12)])
    groups = tf.sensor_groups(s[["x", "y"]].to_numpy(), 200)
    assert len(set(groups)) >= 6
    for g in set(groups):
        xs = s.loc[groups == g, "x"]
        assert xs.max() - xs.min() <= 200


def test_holdouts_and_their_neighbours_are_in_no_fold():
    s = sensors_at((42, 0, 0), (7, 100, 0), (8, 5000, 0), (9, 9000, 0), (10, 13000, 0))
    out = tf.splits(s, {"unimelb_edge": [42]}, n_folds=3, seed=0, radius_m=200)
    out = out.set_index("location_id")
    assert out.loc[42, "fold"] == -1 and out.loc[42, "holdout"] == "unimelb_edge"
    assert out.loc[7, "fold"] == -1 and out.loc[7, "holdout"] == "neighbour"
    assert (out.loc[[8, 9, 10], "fold"] >= 0).all()


def test_splits_are_the_same_for_a_seed():
    rng = np.random.default_rng(1)
    s = sensors_at(*[(i, *rng.uniform(0, 20000, 2)) for i in range(40)])
    a = tf.splits(s, {}, 5, seed=3, radius_m=200)
    b = tf.splits(s, {}, 5, seed=3, radius_m=200)
    assert a.equals(b)
    # Every fold holds out whole groups: no group has rows in two folds.
    assert (a.groupby("group")["fold"].nunique() == 1).all()


def test_event_days_combine_afl_and_logged_events():
    afl = pd.DataFrame(
        {
            "date": ["2026-04-03", "2026-04-03", "2026-04-04"],
            "venue": ["M.C.G.", "Docklands", "Gabba"],
        }
    )
    events = {
        "events": [
            {"start": "2026-11-11T10:30", "kind": "cricket", "venue": "mcg"},
            {"start": "2026-10-31T00:00", "kind": "halloween", "venue": None},
        ]
    }
    days = tf.event_days(afl, {"M.C.G.": "mcg", "Docklands": "marvel"}, events)
    assert days["date_local"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-04-03",
        "2026-11-11",
    ]
    assert days.iloc[0]["venues"] == "marvel,mcg"
    assert days.iloc[1]["kinds"] == "cricket"


def test_reference_quantiles_record_the_spread_and_missing_share():
    ref = tf.reference_quantiles(pd.DataFrame({"a": [0.0, 1.0, 2.0, np.nan]}))
    assert ref["a"]["quantiles"]["0.5"] == 1.0
    assert ref["a"]["null_share"] == 0.25
