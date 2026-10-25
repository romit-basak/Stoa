"""Checks that the stoa_site_build DAG parses and is wired as intended (run in the image, like
tests/dags/test_stoa_pipeline.py)."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(REPO / "dags"))
    assert bag.import_errors == {}
    return bag.get_dag("stoa_site_build")


def test_runs_only_when_triggered(dag):
    assert type(dag.timetable).__name__ == "NullTimetable"
    assert not dag.catchup
    assert dag.max_active_runs == 1


def test_site_choices_are_the_sites_without_counts(dag):
    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())
    others = [s for s, v in cfg["sites"].items() if not v.get("count_sensors")]
    assert dag.params.get_param("site").schema["enum"] == others


def upstream(dag, task_id):
    return set(dag.get_task(task_id).upstream_task_ids)


def test_wiring(dag):
    assert upstream(dag, "site_osm_extract") == {"site_pull_osm"}
    assert upstream(dag, "site_osm_graph") == {"site_osm_extract"}
    assert upstream(dag, "site_osm_segment_features") == {"site_osm_graph"}
    assert upstream(dag, "site_validate_segments") == {"site_osm_segment_features"}
    assert upstream(dag, "site_dvc_push") == {
        "site_validate_segments",
        "site_pull_dem",
        "site_pull_worldcover",
        "site_pull_weather",
        "site_pull_calendars",
        "site_pull_extras",
    }
    assert dag.get_task("site_validate_segments").retries == 0
    assert 99 in dag.get_task("site_dvc_push").skip_on_exit_code
