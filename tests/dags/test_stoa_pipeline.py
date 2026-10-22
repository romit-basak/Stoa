"""Checks that the stoa_pipeline DAG parses and is wired as intended.

Airflow is only installed in the Docker image, so these tests skip in the local .venv. Run
them inside the image:

    docker compose run --rm --no-deps \
        -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////tmp/airflow.db airflow-init \
        bash -c "pip install -q pytest && cd /opt/stoa && python -m pytest tests/dags"

The SQLite override gives the run a throwaway database, so it needs no Postgres.
"""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag

DAGS_DIR = Path(__file__).resolve().parents[2] / "dags"


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS_DIR))
    assert bag.import_errors == {}
    return bag.get_dag("stoa_pipeline")


def test_dag_loads(dag):
    assert dag is not None
    assert dag.max_active_runs == 1
    assert not dag.catchup


def test_has_every_stage(dag):
    assert set(dag.task_ids) == {
        "pull_counts",
        "pull_weather",
        "pull_calendars",
        "pull_osm",
        "pull_static_layers",
        "osm_extract",
        "osm_graph",
        "osm_segment_features",
        "osm_sensor_match",
        "sensor_steepness",
        "join_all",
        "bias_slices",
        "validate",
        "dvc_push",
    }


def test_ingest_tasks_run_in_parallel(dag):
    # The downloads are independent, so none should wait on another.
    for task_id in (
        "pull_counts",
        "pull_weather",
        "pull_calendars",
        "pull_osm",
        "pull_static_layers",
    ):
        assert dag.get_task(task_id).upstream_task_ids == set()


def test_wiring(dag):
    def upstream(task_id):
        return dag.get_task(task_id).upstream_task_ids

    assert upstream("osm_extract") == {"pull_osm"}
    assert upstream("osm_graph") == {"osm_extract"}
    assert upstream("osm_segment_features") == {"osm_graph"}
    assert upstream("osm_sensor_match") == {"osm_segment_features", "pull_counts"}
    assert upstream("sensor_steepness") == {"pull_static_layers", "pull_counts"}
    assert upstream("join_all") == {
        "osm_sensor_match",
        "sensor_steepness",
        "pull_counts",
        "pull_weather",
        "pull_calendars",
    }
    assert upstream("validate") == {"join_all"}
    assert upstream("bias_slices") == {"join_all", "pull_static_layers"}
    assert upstream("dvc_push") == {"validate", "bias_slices"}


def test_every_task_retries(dag):
    for task in dag.tasks:
        if task.task_id != "validate":
            assert task.retries == 2, task.task_id


def test_validate_fails_without_retrying(dag):
    # Bad data fails the same way on every try; retrying would only delay the alert.
    assert dag.get_task("validate").retries == 0


def test_dvc_push_skips_without_credentials(dag):
    # Must match SKIP_EXIT_CODE in src/pipeline/dvc_push.py.
    assert 99 in dag.get_task("dvc_push").skip_on_exit_code
