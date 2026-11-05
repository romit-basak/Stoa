"""Unit tests for the flow model's folds, fold scores, baselines, candidates and selection.

The scoring functions themselves are tested in tests/models/flow/test_evaluate.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.flow import baselines, candidates, select
from src.models.flow.data import final_rows, fold_plan
from src.models.flow.evaluate import log_mae
from src.models.flow.registry import evaluate_gate

# --- folds ----------------------------------------------------------------------------------


def test_fold_plan_never_uses_holdouts(data):
    plan = fold_plan(data.fold, train_rows=10_000, seed=7)
    assert len(plan) == 3
    held = np.flatnonzero(data.fold < 0)
    for k, (train, test) in enumerate(plan):
        assert not np.isin(held, train).any() and not np.isin(held, test).any()
        assert (data.fold[test] == k).all() and (data.fold[train] != k).all()


def test_fold_plan_samples_the_same_rows_every_time(data):
    a = fold_plan(data.fold, train_rows=500, seed=7)
    b = fold_plan(data.fold, train_rows=500, seed=7)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))
    assert all(len(t) == 500 for t, _ in a)


def test_fewer_folds_keeps_the_first_ones(data):
    full = fold_plan(data.fold, train_rows=500, seed=7)
    two = fold_plan(data.fold, train_rows=500, seed=7, n_folds=2)
    assert len(two) == 2
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(full, two))


def test_final_rows_exclude_holdouts(data):
    rows = final_rows(data.fold, 10**6, seed=7)
    assert (data.fold[rows] >= 0).all()


# --- fold scores --------------------------------------------------------------------------


def test_fold_scores_use_her_names_and_the_calibrated_hours(data):
    from src.models.flow.train import fold_scores

    plan = fold_plan(data.fold, 10_000, 7)
    m = fold_scores(data, data.y * 1.1, plan, 3)
    assert {"wape_cv", "mae_cv", "rmse_cv", "log_mae_cv", "wape_cv_cal"} <= set(m)
    assert {f"wape_fold{k}" for k in range(3)} <= set(m)
    assert m["wape_cv"] == pytest.approx(0.1, rel=1e-6)
    assert m["wape_cv_cal"] < 0.01  # one scale per sensor fixes a constant 10% bias


def test_calibration_curve_rescales_the_other_sensors(data):
    rows = data.frame.hourly[["location_id", "ts_utc"]]
    curve = select.calibration_curve(rows, data.y, 2 * data.y, 3, [1, 3], 3, 0)
    by_k = curve.groupby("k")["wape"].mean()
    assert by_k[0] == pytest.approx(1.0)
    assert by_k[1] < 0.01 and by_k[3] < 0.01


# --- baselines ------------------------------------------------------------------------------


def test_baseline_rungs_predict_the_test_folds(data):
    plan = fold_plan(data.fold, 10_000, 7)
    osm = data.frame.osm
    tested = np.concatenate([t for _, t in plan])
    for rung in baselines.RUNGS:
        pred = baselines.oof(rung, data.frame.hourly, osm, data.y, plan)
        assert np.isfinite(pred[tested]).all() and (pred[tested] >= 0).all()
        assert np.isnan(np.delete(pred, tested)).all()


def test_network_terms_help_when_they_drive_the_counts(data):
    # Betweenness sets each synthetic sensor's volume, so rung 1 beats the hour profile alone.
    plan = fold_plan(data.fold, 10_000, 7)
    osm = data.frame.osm
    tested = np.concatenate([t for _, t in plan])
    profile_only = np.full(len(data.y), np.nan, dtype="float32")
    for train, test in plan:
        m = baselines.NetworkBaseline(["junctions_400"], alpha=1e6)
        profile_only[test] = m.fit(data.frame.hourly, osm, train, data.y).predict(
            data.frame.hourly, osm, test
        )
    rung1 = baselines.oof("betweenness", data.frame.hourly, osm, data.y, plan)
    assert log_mae(data.y[tested], rung1[tested]) < log_mae(
        data.y[tested], profile_only[tested]
    )


# --- candidates -----------------------------------------------------------------------------


@pytest.mark.parametrize("family", candidates.FAMILIES)
def test_candidates_fit_and_predict(family, data, fcfg):
    rows = np.flatnonzero(data.fold >= 0)
    x = data.frame.matrix(rows)
    params = candidates.capped(family, fcfg["candidates"][family]["params"], 5)
    model = candidates.make(family, params, 7).fit(x, data.y[rows])
    pred = candidates.predict(model, x)
    assert pred.shape == (len(rows),) and (pred >= 0).all()
    q90 = candidates.make(family, params, 7, quantile=0.9).fit(x, data.y[rows])
    assert (candidates.predict(q90, x) >= 0).all()


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        candidates.make("rf", {}, 7)


def test_capped():
    assert candidates.capped("hgb", {"max_iter": 300}, 5) == {"max_iter": 5}
    assert candidates.capped("lgbm", {"n_estimators": 3}, 5) == {"n_estimators": 3}
    assert candidates.capped("xgb", {"n_estimators": 300}, None) == {
        "n_estimators": 300
    }


# --- selection ------------------------------------------------------------------------------


def test_selection_trades_a_little_accuracy_for_the_fairer_model():
    rank = pd.DataFrame(
        {
            "model": ["lgbm", "hgb", "xgb", "baseline_space_syntax"],
            "kind": ["candidate", "candidate", "candidate", "baseline"],
            "log_mae_cv": [0.500, 0.510, 0.600, 0.700],
        }
    )
    short = select.shortlist(rank, tolerance=0.03)
    assert short == ["lgbm", "hgb"]
    worst = pd.DataFrame({"model": short, "worst_slice_ratio": [2.4, 1.6]})
    assert select.choose(rank, worst) == "hgb"


def test_worst_slice_ignores_thin_groups():
    errors = pd.DataFrame(
        {
            "dimension": ["precinct", "precinct", "time_of_day"],
            "group": ["tiny", "CBD", "night"],
            "n_sensors": [2, 40, 60],
            "wape_vs_overall": [5.0, 1.1, 1.8],
        }
    )
    assert select.worst_slice(errors, 5) == (1.8, "time_of_day=night")


def test_holdout_scores(data, fcfg):
    rows = np.flatnonzero(data.fold >= 0)
    params = candidates.capped("hgb", fcfg["candidates"]["hgb"]["params"], 20)
    model = candidates.make("hgb", params, 7).fit(data.frame.matrix(rows), data.y[rows])
    table, holdout = select.holdout_scores(data, model, 3)
    assert set(table["precinct"]) == {"carlton"}
    assert set(table["calibrated_on"]) == {
        0,
        1,
    }  # two sensors: calibrate on one, score one
    assert set(holdout) == {"log_mae_holdout", "wape_holdout"}
    assert 0 < holdout["wape_holdout"] < 1


# --- gate -----------------------------------------------------------------------------------

GATES = {
    "log_mae_improvement": 0.15,
    "coverage80": [0.75, 0.85],
    "max_worst_slice_ratio": None,
}


def test_gate_passes():
    ok, reasons, imp = evaluate_gate(
        {"log_mae_cv": 0.4, "coverage80_cv": 0.8}, {"log_mae_cv": 0.5}, GATES
    )
    assert ok and not reasons and imp == pytest.approx(0.2)


@pytest.mark.parametrize(
    "selected, baseline, why",
    [
        (
            {"log_mae_cv": 0.45, "coverage80_cv": 0.8},
            {"log_mae_cv": 0.5},
            "below Space Syntax",
        ),
        (
            {"log_mae_cv": 0.4, "coverage80_cv": 0.9},
            {"log_mae_cv": 0.5},
            "coverage80_cv",
        ),
        ({"log_mae_cv": 0.4, "coverage80_cv": 0.8}, None, "no baseline"),
    ],
)
def test_gate_fails(selected, baseline, why):
    ok, reasons, _ = evaluate_gate(selected, baseline, GATES)
    assert not ok and any(why in r for r in reasons)


def test_gate_can_block_on_bias():
    gates = {**GATES, "max_worst_slice_ratio": 2.0}
    selected = {"log_mae_cv": 0.4, "coverage80_cv": 0.8, "worst_slice_ratio": 2.5}
    ok, reasons, _ = evaluate_gate(selected, {"log_mae_cv": 0.5}, gates)
    assert not ok and "worst slice" in reasons[0]
