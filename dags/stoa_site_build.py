"""Build a site's walking graph and segment features from public data (no counts needed).

Trigger it by hand with `site` set (neu_boston for the demonstration, nyc for the cross-city
transfer test). It pulls the site's OSM extract, DEM, land cover, weather and calendars, plus
any extra layers the site has (building heights and imagery for NEU Boston), runs the OSM
feature stages, checks the segment table and saves it as a new DVC version. Melbourne is built
by stoa_pipeline, which also needs its counts.

The OSM extract is pinned per site, as in stoa_pipeline; the OSM stages skip themselves when
nothing they read has changed (--if-stale).
"""

from datetime import UTC, datetime, timedelta

from airflow.sdk import Param, dag, task

STOA = "cd /opt/stoa && python -m"
SITE = "{{ params.site }}"


@dag(
    dag_id="stoa_site_build",
    description="Build a site's walking graph and segment features",
    start_date=datetime(2026, 10, 25, tzinfo=UTC),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    params={
        "site": Param(
            "neu_boston",
            type="string",
            enum=["neu_boston", "nyc"],
            description="Site to build (configs/data.yaml, sites).",
        )
    },
    tags=["stoa", "site-build"],
)
def stoa_site_build():
    @task.bash
    def site_pull_osm():
        return f"{STOA} src.ingest.osm --site {SITE}"

    @task.bash
    def site_pull_dem():
        return f"{STOA} src.ingest.dem --site {SITE}"

    @task.bash
    def site_pull_worldcover():
        return f"{STOA} src.ingest.worldcover --site {SITE}"

    @task.bash
    def site_pull_weather():
        return f"{STOA} src.ingest.weather --site {SITE}"

    @task.bash
    def site_pull_calendars():
        return f"{STOA} src.ingest.calendars"

    @task.bash
    def site_pull_extras():
        # Building heights, imagery: whatever sites.<site>.extra_ingest lists.
        return f"{STOA} src.pipeline.site_extras --site {SITE}"

    @task.bash
    def site_osm_extract():
        return f"{STOA} src.features.osm --site {SITE} --only extract --if-stale"

    @task.bash
    def site_osm_graph():
        return f"{STOA} src.features.osm --site {SITE} --only graph --if-stale"

    @task.bash
    def site_osm_segment_features():
        return f"{STOA} src.features.osm --site {SITE} --only features --if-stale"

    @task.bash(retries=0)
    def site_validate_segments():
        # Fails on any error-level check, so a broken segment table is never versioned.
        return f"{STOA} src.pipeline.validate_segments --site {SITE}"

    @task.bash(skip_on_exit_code=99)
    def site_dvc_push():
        return f"{STOA} src.pipeline.dvc_push"

    extract = site_osm_extract()
    features = site_osm_segment_features()
    site_pull_osm() >> extract >> site_osm_graph() >> features
    checked = site_validate_segments()
    features >> checked
    pulls = [
        site_pull_dem(),
        site_pull_worldcover(),
        site_pull_weather(),
        site_pull_calendars(),
        site_pull_extras(),
    ]
    [checked, *pulls] >> site_dvc_push()


stoa_site_build()
