"""Tests for src/models/flow/evaluate.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.flow.evaluate import (
    check_gates,
    interval_coverage,
    mae_log,
    wape,
)


def test_wape_perfect():
    y = np.array([10.0, 20.0, 30.0])
    assert wape(y, y) == pytest.approx(0.0)


def test_wape_double():
    y = np.array([10.0, 20.0, 30.0])
    # Predicting 2x → sum of |e| = sum(y), ratio = 1
    assert wape(y, 2 * y) == pytest.approx(1.0)


def test_wape_zero_true_is_nan():
    assert np.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 1.0])))


def test_mae_log_perfect():
    y = np.array([0.0, 10.0, 100.0])
    assert mae_log(y, y) == pytest.approx(0.0)


def test_mae_log_clamps_negative_preds():
    y = np.array([10.0, 10.0])
    # Negative predictions clamped to 0 before log1p
    result = mae_log(y, np.array([-5.0, 10.0]))
    assert result > 0.0


def test_interval_coverage_all_in():
    y = np.array([5.0, 10.0, 15.0])
    q10 = np.array([4.0, 9.0, 14.0])
    q90 = np.array([6.0, 11.0, 16.0])
    assert interval_coverage(y, q10, q90) == pytest.approx(1.0)


def test_interval_coverage_none_in():
    y = np.array([5.0, 10.0, 15.0])
    q10 = np.array([6.0, 11.0, 16.0])
    q90 = np.array([7.0, 12.0, 17.0])
    assert interval_coverage(y, q10, q90) == pytest.approx(0.0)


def test_interval_coverage_half():
    y = np.array([5.0, 10.0])
    q10 = np.array([4.0, 11.0])
    q90 = np.array([6.0, 12.0])
    assert interval_coverage(y, q10, q90) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def test_check_gates_passes():
    cfg = {
        "gates": {
            "mae_log_improvement_pct": 15.0,
            "interval_coverage_low": 0.75,
            "interval_coverage_high": 0.85,
        }
    }
    # 20% improvement, 80% coverage → pass
    passed, msg = check_gates(0.80, 1.00, 0.80, cfg)
    assert passed
    assert "passed" in msg.lower()


def test_check_gates_fails_improvement():
    cfg = {
        "gates": {
            "mae_log_improvement_pct": 15.0,
            "interval_coverage_low": 0.75,
            "interval_coverage_high": 0.85,
        }
    }
    # Only 10% improvement → fail
    passed, _ = check_gates(0.90, 1.00, 0.80, cfg)
    assert not passed


def test_check_gates_fails_coverage():
    cfg = {
        "gates": {
            "mae_log_improvement_pct": 15.0,
            "interval_coverage_low": 0.75,
            "interval_coverage_high": 0.85,
        }
    }
    # Good improvement but coverage too low
    passed, _ = check_gates(0.80, 1.00, 0.60, cfg)
    assert not passed
