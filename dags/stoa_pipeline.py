"""Stoa data pipeline: pull the Melbourne sources, build OSM features, then (TODO) join,
validate and version the result.

Each ingest task runs an existing `src.ingest` module as-is, from the repo mounted at
/opt/stoa. Tasks marked TODO are placeholders for teammates' code; replace the body,
keep the task name, and the wiring at the bottom stays the same.
"""

from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

STOA = "cd /opt/stoa && python -m"


@dag(
    dag_id="stoa_pipeline",
    start_date=datetime(2026, 10, 1, tzinfo=UTC),
    # The portal adds new counts daily and keeps only ~2 years, so pull daily and never
    # miss a window. catchup=False: a missed day is covered by the next run's merge.
    schedule="@daily",
    catchup=False,
    max_active_runs=1,  # two runs merging into the same parquet at once would clash
    tags=["stoa", "melbourne"],
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
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
    def pull_osm():
        return f"{STOA} src.ingest.osm --site melbourne"

    # --- OSM features: one task per stage so each shows up on the Gantt chart ---

    @task.bash
    def osm_extract():
        return f"{STOA} src.features.osm --site melbourne --only extract"

    @task.bash
    def osm_graph():
        return f"{STOA} src.features.osm --site melbourne --only graph"

    @task.bash
    def osm_segment_features():
        return f"{STOA} src.features.osm --site melbourne --only features"

    @task.bash
    def osm_sensor_match():
        return f"{STOA} src.features.osm --site melbourne --only sensors"

    # --- Placeholders (TODO) ---

    @task
    def join_all():
        print("TODO: join counts + sensor_osm_features + weather + calendars")

    @task
    def validate():
        print("TODO: schema checks, completeness per sensor, zero/stuck-value checks")

    @task
    def dvc_push():
        print("TODO: dvc add data/ && dvc push")

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

    joined = join_all()
    [sensors, counts, pull_weather(), pull_calendars()] >> joined
    joined >> validate() >> dvc_push()


stoa_pipeline()
