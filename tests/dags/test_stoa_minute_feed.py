"""Checks that the stoa_minute_feed DAG parses and polls as intended (run in the image, like
tests/dags/test_stoa_pipeline.py)."""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS_DIR))
    assert bag.import_errors == {}
    return bag.get_dag("stoa_minute_feed")


def test_polls_every_15_minutes_without_catching_up(dag):
    assert dag is not None
    assert dag.timetable.expression == "*/15 * * * *"
    assert not dag.catchup
    assert dag.max_active_runs == 1


def test_one_poll_task_that_skips_without_credentials(dag):
    assert [t.task_id for t in dag.tasks] == ["poll_minute_feed"]
    task = dag.get_task("poll_minute_feed")
    assert 99 in task.skip_on_exit_code
