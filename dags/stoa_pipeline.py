"""Stoa data pipeline: pull the Melbourne sources, build OSM features, join the hourly table,
then (TODO) validate it and save it as a new DVC version.

Each ingest task runs an existing `src.ingest` module as-is, from the repo mounted at
/opt/stoa. Tasks marked TODO are placeholders for teammates' code; replace the body,
keep the task name, and the wiring at the bottom stays the same.

The OSM extract is pinned: a daily run keeps the file it has, and the OSM stages skip
themselves (--if-stale) unless the extract, the feature config or code, the sensor
locations or the sensor overrides changed. To move to a newer extract, trigger the DAG with
refresh_osm = true. That renumbers graph segments, so re-check the sensor overrides
afterwards; osm_sensor_match fails until stale ones are fixed.
"""

import os
from datetime import UTC, datetime, timedelta

from airflow.providers.smtp.notifications.smtp import SmtpNotifier
from airflow.sdk import Param, dag, task

STOA = "cd /opt/stoa && python -m"

# Email the team when a task fails after its last retry. Recipients and the Gmail login come
# from .env via docker-compose.yaml; with STOA_ALERT_EMAILS unset, alerts are off.
ALERT_EMAILS = [
    e.strip() for e in os.environ.get("STOA_ALERT_EMAILS", "").split(",") if e.strip()
]
on_failure_alert = (
    [
        SmtpNotifier(
            to=ALERT_EMAILS,
            from_email=os.environ.get("STOA_SMTP_USER"),
            subject="[Stoa] {{ dag.dag_id }}: {{ ti.task_id }} failed",
            html_content=(
                "<p>Task <b>{{ ti.task_id }}</b> in <b>{{ dag.dag_id }}</b> failed after "
                "{{ ti.try_number }} tries.</p>"
                "<p>Run: {{ run_id }}</p>"
                '<p><a href="{{ ti.log_url }}">Open the log</a></p>'
            ),
        )
    ]
    if ALERT_EMAILS
    else []
)


@dag(
    dag_id="stoa_pipeline",
    start_date=datetime(2026, 10, 1, tzinfo=UTC),
    # The portal adds new counts daily and keeps only ~2 years, so pull daily and never
    # miss a window. catchup=False: a missed day is covered by the next run's merge.
    schedule="@daily",
    catchup=False,
    max_active_runs=1,  # two runs merging into the same parquet at once would clash
    tags=["stoa", "melbourne"],
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "on_failure_callback": on_failure_alert,
    },
    params={
        "refresh_osm": Param(
            False,
            type="boolean",
            description="Download the latest OSM extract (renumbers graph segments).",
        )
    },
)
def stoa_pipeline():
    # --- Ingest: independent downloads, so they run in parallel ---

    @task.bash
    def pull_counts():
        # Hourly counts (merged into the rows already held) and the sensor locations,
        # which sensor matching needs.
        return (
            f"{STOA} src.ingest.melbourne_portal --only "
            "pedestrian-counting-system-monthly-counts-per-hour "
            "pedestrian-counting-system-sensor-locations"
        )

    @task.bash
    def pull_weather():
        return f"{STOA} src.ingest.weather --site melbourne"

    @task.bash
    def pull_calendars():
        return f"{STOA} src.ingest.calendars"

    @task.bash
    def pull_terrain():
        # Footpath-steepness survey and the GA 5 m DEM. Both are fixed snapshots, so each is
        # fetched only when it isn't held yet (a fresh clone without `dvc pull`).
        raw = "/opt/stoa/data/raw"
        return (
            f"test -f {raw}/melbourne_portal/footpath-steepness.geojson || "
            f"{STOA} src.ingest.melbourne_portal --only footpath-steepness; "
            f"test -f {raw}/dem/melbourne_ga_lidar_5m.tif || "
            f"{STOA} src.ingest.dem --site melbourne"
        )

    @task.bash
    def pull_osm(params=None):
        refresh = " --refresh" if params and params.get("refresh_osm") else ""
        return f"{STOA} src.ingest.osm --site melbourne{refresh}"

    # --- OSM features: one task per stage so each shows up on the Gantt chart ---

    @task.bash
    def osm_extract():
        return f"{STOA} src.features.osm --site melbourne --only extract --if-stale"

    @task.bash
    def osm_graph():
        return f"{STOA} src.features.osm --site melbourne --only graph --if-stale"

    @task.bash
    def osm_segment_features():
        return f"{STOA} src.features.osm --site melbourne --only features --if-stale"

    @task.bash
    def osm_sensor_match():
        return f"{STOA} src.features.osm --site melbourne --only sensors --if-stale"

    @task.bash
    def sensor_steepness():
        # Footpath slope and elevation per sensor (from Shubham Hadawle's join notebook).
        return f"{STOA} src.features.sensor_steepness"

    # --- Join, validate (TODO), then version ---

    @task.bash
    def join_all():
        # Hourly training table: counts + time + weather + calendars + sensor match + slope.
        return f"{STOA} src.features.counts_hourly"

    @task
    def validate():
        print("TODO: schema checks, completeness per sensor, zero/stuck-value checks")

    # Exit code 99 (src.pipeline.dvc_push.SKIP_EXIT_CODE) means there are no credentials
    # for the bucket, as on a fresh clone outside the team: skip instead of failing the run.
    @task.bash(skip_on_exit_code=99)
    def dvc_push():
        # Runs after validate, so only data that passed the checks becomes a version. It
        # uploads the data; committing the updated data/*.dvc pointers is left to a person.
        return f"{STOA} src.pipeline.dvc_push"

    # --- Wiring ---

    counts = pull_counts()
    # Keep a handle on each stage: `a >> b >> c` returns c, so chaining onto that result
    # would attach to the last stage, not the first.
    extract = osm_extract()
    osm_features = osm_segment_features()
    pull_osm() >> extract >> osm_graph() >> osm_features

    # Sensor matching needs both the segment features and the sensor locations.
    sensors = osm_sensor_match()
    [osm_features, counts] >> sensors

    # Slope matching needs the sensor locations and the terrain files.
    steepness = sensor_steepness()
    [pull_terrain(), counts] >> steepness

    joined = join_all()
    [sensors, steepness, counts, pull_weather(), pull_calendars()] >> joined
    joined >> validate() >> dvc_push()


stoa_pipeline()
