"""Checks that the stoa_train DAG parses and runs its stages in order (run in the image)."""

import itertools
from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS_DIR))
    assert bag.import_errors == {}
    return bag.get_dag("stoa_train")


def test_runs_monthly_without_catching_up(dag):
    assert not dag.catchup and dag.max_active_runs == 1
    assert "tune" in dag.params


def test_stages_run_in_order(dag):
    order = [
        "train_candidates",
        "select_model",
        "mitigate_bias",
        "check_gates",
        "register_model",
    ]
    assert set(dag.task_ids) == set(order)
    for a, b in itertools.pairwise(order):
        assert dag.get_task(a).downstream_task_ids == {b}


def test_daily_pipeline_does_not_train():
    bag = DagBag(dag_folder=str(DAGS_DIR))
    assert not any("flow" in t for t in bag.get_dag("stoa_pipeline").task_ids)
