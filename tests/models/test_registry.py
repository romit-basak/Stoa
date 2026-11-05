"""Tests for src/models/flow/registry.py — all MLflow calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.flow.registry import find_best_run


def _make_run(
    run_id: str, mae_log: float, coverage: float, ss_mae: float | None = None
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.status = "FINISHED"
    m: dict[str, float] = {"mae_log": mae_log, "interval_coverage": coverage}
    if ss_mae is not None:
        m["space_syntax_mae_log"] = ss_mae
    run.data.metrics = m
    return run


@patch("src.models.flow.registry.mlflow")
def test_find_best_run_raises_when_none_qualify(mock_mlflow):
    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    # Run with bad coverage (too low)
    mock_client.search_runs.return_value = [
        _make_run("run1", mae_log=0.80, coverage=0.50)
    ]
    with pytest.raises(ValueError, match="No finished run"):
        find_best_run("stoa_flow_model", 15.0, 0.75, 0.85)


@patch("src.models.flow.registry.mlflow")
def test_find_best_run_returns_qualifying_run(mock_mlflow):
    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    # This run has good coverage; no ss_mae so improvement gate is skipped
    mock_client.search_runs.return_value = [
        _make_run("run_good", mae_log=0.80, coverage=0.80)
    ]
    run_id = find_best_run("stoa_flow_model", 15.0, 0.75, 0.85)
    assert run_id == "run_good"


@patch("src.models.flow.registry.mlflow")
def test_find_best_run_improvement_gate_fails(mock_mlflow):
    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    # Only 10% improvement (need 15%)
    mock_client.search_runs.return_value = [
        _make_run("run_bad", mae_log=0.90, coverage=0.80, ss_mae=1.00)
    ]
    with pytest.raises(ValueError):
        find_best_run("stoa_flow_model", 15.0, 0.75, 0.85)


@patch("src.models.flow.registry.mlflow")
def test_find_best_run_improvement_gate_passes(mock_mlflow):
    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="1")
    # 20% improvement, good coverage
    mock_client.search_runs.return_value = [
        _make_run("run_ok", mae_log=0.80, coverage=0.80, ss_mae=1.00)
    ]
    run_id = find_best_run("stoa_flow_model", 15.0, 0.75, 0.85)
    assert run_id == "run_ok"
