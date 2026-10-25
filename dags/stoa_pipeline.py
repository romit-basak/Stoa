"""Stoa data pipeline: pull the Melbourne sources, build OSM features, join the hourly table,
report how much data each bias slice has, validate it, and save it as a new DVC version.

Each task runs an existing `src` module as-is, from the repo mounted at /opt/stoa.

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
    # Counts and weather arrive daily, and validation should catch a dead sensor or a stale
    # feed within a day. catchup=False: each run re-fetches from two days before the newest
    # held date, so a missed day is covered by the next run's merge.
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

    # Exit 99 (no bucket credentials) skips it, as for dvc_push.
    @task.bash(skip_on_exit_code=99)
    def pull_minute_feed():
        # The per-minute archive (stoa_minute_feed writes it to the bucket) into data/raw,
        # where DVC versions it and validate checks it.
        return f"{STOA} src.ingest.minute_feed --sync"

    @task.bash
    def pull_calendars():
        return f"{STOA} src.ingest.calendars"

    @task.bash
    def pull_static_layers():
        # Layers that don't change day to day, so each is fetched only when it isn't held yet
        # (a fresh clone without `dvc pull`): the footpath-steepness survey and the GA 5 m DEM
        # (slope), the CLUE areas (precincts) and AFL crowds (match days), for bias slicing.
        # Each step is `{ test -f FILE || { fetch; }; }`, and the steps are joined with &&.
        # The braces matter: || and && have equal precedence, so `test || cd && python` would
        # run python even when the file is held (from the wrong folder, so it fails), and a
        # failed first fetch must stop the task rather than be hidden by the second.
        raw = "/opt/stoa/data/raw"
        portal = f"{raw}/melbourne_portal"
        clue = "blocks-for-census-of-land-use-and-employment-clue"
        steps = [
            (
                f"{portal}/footpath-steepness.geojson",
                f"{STOA} src.ingest.melbourne_portal --only footpath-steepness",
            ),
            (
                f"{portal}/{clue}.geojson",
                f"{STOA} src.ingest.melbourne_portal --only {clue}",
            ),
            (
                f"{raw}/dem/melbourne_ga_lidar_5m.tif",
                f"{STOA} src.ingest.dem --site melbourne",
            ),
            (
                f"{raw}/events/afl_attendance.csv",
                f"{STOA} src.ingest.afl_attendance",
            ),
        ]
        return " && ".join(
            f"{{ test -f {path} || {{ {cmd}; }}; }}" for path, cmd in steps
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
    def osm_shade():
        # Building shade per segment and sun position (src/features/shade.py).
        return f"{STOA} src.features.osm --site melbourne --only shade --if-stale"

    @task.bash
    def osm_sensor_match():
        return f"{STOA} src.features.osm --site melbourne --only sensors --if-stale"

    @task.bash
    def sensor_steepness():
        # Footpath slope and elevation per sensor (from Shubham Hadawle's join notebook).
        return f"{STOA} src.features.sensor_steepness"

    # --- Join, validate, then version ---

    @task.bash
    def join_all():
        # Hourly training table: counts + time + weather + calendars + sensor match + slope.
        return f"{STOA} src.features.counts_hourly"

    @task.bash
    def bias_slices():
        # Bias step 1: how much data each slice (place, time, weather, era...) has against a
        # reference. Needs no model. See docs/bias/bias_slicing.md.
        return f"{STOA} src.bias.representation"

    @task.bash
    def bias_errors():
        # Bias steps 2-3: a baseline's error per slice on held-out sensors, with and without
        # reweighting. About 15-20 minutes, so it reruns only weekly or when its settings change.
        return f"{STOA} src.bias.slice_errors --if-older-than 7"

    @task.bash
    def training_frame():
        # The location splits, event days and feature spec every model trains from
        # (src/features/training_frame.py). About 10 s.
        return f"{STOA} src.features.training_frame"

    # No retries: bad data fails the same way every time, so retrying would only delay the
    # failure email. Checks and thresholds: configs/validation.yaml, docs/pipeline/validation.md.
    @task.bash(retries=0)
    def validate():
        # Schema, Great Expectations suite, anomalies and drift on the hourly table. Fails on
        # any error-level check, so dvc_push never versions bad data.
        return f"{STOA} src.pipeline.validate"

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
    built = osm_graph()
    pull_osm() >> extract >> built >> osm_features
    shaded = osm_shade()
    built >> shaded

    # Sensor matching needs both the segment features and the sensor locations.
    sensors = osm_sensor_match()
    [osm_features, counts] >> sensors

    # Slope matching needs the sensor locations and the terrain files.
    static = pull_static_layers()
    steepness = sensor_steepness()
    [static, counts] >> steepness

    joined = join_all()
    [sensors, steepness, counts, pull_weather(), pull_calendars()] >> joined

    # Bias slicing runs after validation, not beside it. Both load the whole hourly table
    # (about 3 GB each at 7.4M rows), and side by side they overran Docker Desktop's default
    # 8 GB: validate was killed (exit -9). This order also keeps the bias reports to data that
    # passed the checks. Only data that went through all three is versioned.
    checked = validate()
    [joined, pull_minute_feed()] >> checked
    # The training frame loads the table too (about 2 GB), so it also takes its turn.
    frame = training_frame()
    [checked, shaded] >> frame
    slices = bias_slices()
    [frame, static] >> slices
    errors = bias_errors()
    slices >> errors
    [checked, errors] >> dvc_push()


stoa_pipeline()
