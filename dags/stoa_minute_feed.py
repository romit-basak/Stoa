"""Archive the City of Melbourne's per-minute pedestrian counts every 15 minutes.

The portal keeps counts from 00:55 Melbourne time until a reset at about 02:00 the next morning,
so a poll has to land between midnight and that reset, or the end of the day before and the new
day's first hour are lost. Each run fetches the feed and
merges it into that day's file in the project bucket (src/ingest/minute_feed.py); several
teammates' machines can run this DAG at once without overwriting each other. Without bucket
credentials the task is skipped, not failed.

Missed runs aren't caught up (catchup=False): the next run reads the whole day so far anyway.
No failure email: a failed poll is covered by the next one, and gaps that matter are reported by
the daily stoa_pipeline validation.
"""

from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

STOA = "cd /opt/stoa && python -m"


@dag(
    dag_id="stoa_minute_feed",
    description="Archive the per-minute pedestrian counts into the project bucket",
    start_date=datetime(2026, 10, 25, tzinfo=UTC),
    schedule="*/15 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    dagrun_timeout=timedelta(minutes=10),
    tags=["stoa", "melbourne", "ingest"],
)
def stoa_minute_feed():
    @task.bash(skip_on_exit_code=99)
    def poll_minute_feed():
        return f"{STOA} src.ingest.minute_feed --bucket"

    poll_minute_feed()


stoa_minute_feed()
