import numpy as np
import pandas as pd
import pytest

from src.models.flow.evaluate import (
    calibrated_cv_metrics,
    coverage,
    cv_coverage,
    cv_metrics,
    log_mae,
    score,
    score_parts,
    scored_mask,
    standard_parts,
)


def rows_for(sensors) -> pd.DataFrame:
    return pd.DataFrame({"location_id": sensors})


def test_perfect_predictions_score_zero():
    y = np.array([0.0, 10, 100, 1000, 5, 50])
    s = score(rows_for([1, 1, 2, 2, 3, 3]), y, y.copy())
    assert s["log_mae"] == 0 and s["wape"] == 0 and s["mae"] == 0 and s["bias"] == 0
    assert s["sensor_rank_corr"] == pytest.approx(1.0)
    assert (s["rows"], s["sensors"]) == (6, 3)


def test_predicting_double_costs_about_log_2():
    y = np.full(100, 1000.0)
    assert log_mae(y, 2 * y) == pytest.approx(np.log(2), abs=1e-3)


def test_one_bad_sensor_shows_as_the_worst():
    sensors = np.repeat([1, 2, 3, 4], 5)
    y = np.repeat([10.0, 20, 30, 40], 5)
    pred = np.where(sensors == 4, 80.0, y)  # sensor 4 predicted twice as busy
    s = score(rows_for(sensors), y, pred)
    assert s["sensor_wape_worst"] == pytest.approx(1.0)
    assert s["sensor_wape_median"] == 0
    # The order of sensors is still right.
    assert s["sensor_rank_corr"] == pytest.approx(1.0)


def test_a_model_that_predicts_the_same_everywhere_gets_no_ranking_credit():
    # Busier sensors have more hours, so ranking by totals would score this model 1.0.
    sensors = np.repeat([1, 2, 3], [2, 4, 6])
    y = np.repeat([10.0, 20, 30], [2, 4, 6])
    s = score(rows_for(sensors), y, np.full(len(y), 15.0))
    assert np.isnan(s["sensor_rank_corr"])


def test_bad_input_is_refused():
    one = rows_for([1])
    with pytest.raises(ValueError):
        score(one, np.array([1.0]), np.array([np.nan]))
    with pytest.raises(ValueError):
        score(one, np.array([1.0]), np.array([-1.0]))
    with pytest.raises(ValueError):
        score(rows_for([1, 2]), np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        score(rows_for([]), np.array([]), np.array([]))


def test_calibration_window_hours_are_not_scored():
    ts = pd.Series(
        pd.to_datetime(
            ["2026-01-01", "2026-01-20", "2026-02-15", "2026-01-10", "2026-03-01"],
            utc=True,
        )
    )
    mask = scored_mask(np.array([1, 1, 1, 2, 2]), ts, 28)
    assert mask.tolist() == [False, False, True, False, True]


def test_event_days_and_eras_are_scored_apart():
    rows = pd.DataFrame(
        {
            "location_id": [1, 1, 2, 2],
            "date_local": pd.to_datetime(
                ["2019-05-01", "2020-06-01", "2026-01-03", "2026-01-04"]
            ).astype("datetime64[us]"),
        }
    )
    eras = {
        "pre_covid": "2009-01-01",
        "covid": "2020-03-16",
        "post_metro_tunnel": "2025-11-30",
    }
    parts = standard_parts(rows, pd.Series(pd.to_datetime(["2026-01-03"])), eras)
    assert parts["event_days"].tolist() == [False, False, True, False]
    assert parts["other_days"].tolist() == [True, True, False, True]
    assert parts["era_pre_covid"].tolist() == [True, False, False, False]
    assert parts["era_covid"].tolist() == [False, True, False, False]
    assert parts["era_post_metro_tunnel"].tolist() == [False, False, True, True]


def test_an_empty_part_reports_zero_rows():
    y = np.array([5.0, 6.0])
    parts = {"all": np.array([True, True]), "none": np.array([False, False])}
    out = score_parts(rows_for([1, 2]), y, y, parts)
    assert out["none"] == {"rows": 0, "sensors": 0}
    assert out["all"]["rows"] == 2


def test_rmse_is_the_root_of_the_mean_squared_error():
    s = score(rows_for([1, 2]), np.array([1.0, 1.0]), np.array([4.0, 5.0]))
    assert s["rmse"] == pytest.approx(np.sqrt(12.5))


def test_each_fold_is_scored_and_the_folds_are_averaged():
    sensors = np.repeat([1, 2, 3, 4, 5, 6], 2)
    y = np.repeat([10.0, 20, 30, 40, 50, 60], 2)
    folds = np.repeat([0, 0, 0, 1, 1, 1], 2)
    pred = np.where(folds == 1, 2 * y, y)  # fold 0 perfect, fold 1 twice too busy
    m = cv_metrics(rows_for(sensors), y, pred, folds)
    assert m["wape_fold0"] == 0 and m["wape_fold1"] == pytest.approx(1.0)
    assert m["wape_cv"] == pytest.approx(0.5)
    expected = {
        f"{n}_{p}"
        for n in ("wape", "log_mae", "mae", "rmse", "bias", "sensor_rank_corr")
        for p in ("fold0", "fold1", "cv")
    }
    assert set(m) == expected


def test_holdout_rows_are_refused_by_the_fold_scores():
    y = np.array([5.0, 6.0])
    with pytest.raises(ValueError):
        cv_metrics(rows_for([1, 2]), y, y, np.array([0, -1]))


def test_coverage_counts_the_hours_inside_the_interval():
    y = np.array([5.0, 10, 15, 20])
    assert coverage(y, y - 1, y + 1) == 1.0
    assert coverage(y, np.full(4, 8.0), np.full(4, 16.0)) == 0.5
    assert coverage(y, y + 1, y + 2) == 0.0


def test_coverage_refuses_bad_input():
    with pytest.raises(ValueError):
        coverage(np.array([1.0]), np.array([0.0, 0.0]), np.array([2.0]))
    with pytest.raises(ValueError):
        coverage(np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError):
        coverage(np.array([1.0]), np.array([np.nan]), np.array([2.0]))


def test_coverage_is_scored_per_fold_and_averaged():
    y = np.array([5.0, 5, 5, 5])
    low = np.array([4.0, 4, 6, 4])
    high = np.full(4, 6.0)
    m = cv_coverage(y, low, high, np.array([0, 0, 1, 1]))
    assert m == {
        "coverage80_fold0": 1.0,
        "coverage80_fold1": 0.5,
        "coverage80_cv": 0.75,
    }
    with pytest.raises(ValueError):
        cv_coverage(y, low, high, np.array([0, 0, 1, -1]))


def test_calibration_fixes_a_constant_scale_and_scores_only_later_hours():
    ts = pd.Series(
        pd.to_datetime(
            ["2026-01-01", "2026-01-10", "2026-02-15", "2026-03-01"] * 2, utc=True
        )
    )
    rows = pd.DataFrame({"location_id": np.repeat([1, 2], 4), "ts_utc": ts})
    y = np.array([10.0, 20, 30, 40, 50, 60, 70, 80])
    folds = np.repeat([0, 1], 4)
    m = calibrated_cv_metrics(rows, y, 3 * y, folds, 28)
    assert m["wape_cv_cal"] == pytest.approx(0.0, abs=1e-12)
    assert m["log_mae_cv_cal"] == pytest.approx(0.0, abs=1e-12)
    assert set(m) == {f"{k}_cal" for k in cv_metrics(rows, y, y, folds)}
    # The window hours are left out: two later hours per sensor are scored.
    assert m["wape_fold0_cal"] == 0.0
    with pytest.raises(ValueError):
        calibrated_cv_metrics(rows, y[:-1], 3 * y[:-1], folds[:-1], 28)
