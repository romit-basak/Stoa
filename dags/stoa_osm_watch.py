"""Once a week, report OSM path changes near the count sensors since the pinned extract.

Downloads the latest extract to a temporary folder (never replacing data/raw/osm), compares the
walkable ways with the pinned ones and warns about ways added, removed, re-routed or retagged
within 100 m of a sensor (src/pipeline/osm_watch.py). A change that matters is the cue to
refresh the extract on purpose: stoa_pipeline with refresh_osm, then re-check the sensor
overrides.
"""

from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

STOA = "cd /opt/stoa && python -m"


@dag(
    dag_id="stoa_osm_watch",
    description="Weekly report of OSM path changes near the count sensors",
    start_date=datetime(2026, 10, 25, tzinfo=UTC),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    tags=["stoa", "monitoring"],
)
def stoa_osm_watch():
    @task.bash
    def watch_osm_changes():
        return f"{STOA} src.pipeline.osm_watch"

    watch_osm_changes()


stoa_osm_watch()
