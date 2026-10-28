"""Checks that the stoa_osm_watch DAG parses and runs weekly (run in the image)."""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"


def test_runs_weekly_without_catching_up():
    bag = DagBag(dag_folder=str(DAGS_DIR))
    assert bag.import_errors == {}
    dag = bag.get_dag("stoa_osm_watch")
    assert [t.task_id for t in dag.tasks] == ["watch_osm_changes"]
    assert not dag.catchup and dag.max_active_runs == 1
