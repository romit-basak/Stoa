"""Train, select, gate and register the flow model (src/models/flow/).

Monthly, and on demand. It trains on the hourly table stoa_pipeline last built, so trigger it
after a good stoa_pipeline run. Runs and models go to the MLflow service in docker-compose.yaml
(MLFLOW_TRACKING_URI). See docs/models/flow_model.md.

- train_candidates: every candidate architecture and the baseline ladder on the location
  folds (with tune = true, an Optuna search per family first);
- select_model: the bias check on the shortlist, the interval models, the final fit, and the
  holdout precincts;
- check_gates: fails the run when the rich model doesn't beat Space Syntax by 15% or its 80%
  intervals miss 75-85% coverage, so nothing below the bar is registered;
- register_model: registers both variants and moves `champion` to them (the version it
  replaces becomes `previous`, for `src.models.flow.registry --rollback`).
"""

import os
from datetime import UTC, datetime

from airflow.providers.smtp.notifications.smtp import SmtpNotifier
from airflow.sdk import Param, dag, task

STOA = "cd /opt/stoa && python -m"

# The same alert as stoa_pipeline: email the team when a task fails.
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
    dag_id="stoa_train",
    description="Train, select, gate and register the flow model",
    start_date=datetime(2026, 11, 1, tzinfo=UTC),
    # CLAUDE.md, "Retrain": monthly on new counts.
    schedule="@monthly",
    catchup=False,
    max_active_runs=1,
    tags=["stoa", "models"],
    # No retries: training fails the same way twice, and a failed gate is a result.
    default_args={"retries": 0, "on_failure_callback": on_failure_alert},
    params={
        "tune": Param(
            False,
            type="boolean",
            description="Run an Optuna search per candidate family before training.",
        )
    },
)
def stoa_train():
    @task.bash
    def train_candidates(params=None):
        tune = " --tune" if params and params.get("tune") else ""
        return f"{STOA} src.models.flow.train --variant both{tune}"

    @task.bash
    def select_model():
        return f"{STOA} src.models.flow.select --variant both"

    @task.bash
    def mitigate_bias():
        # Retrains the selected model with reweighted rows and calibrates it, scored on the
        # same slices (src/models/flow/mitigate.py; about 20 minutes for both variants).
        return f"{STOA} src.models.flow.mitigate --variant both"

    @task.bash
    def check_gates():
        return f"{STOA} src.models.flow.registry --check"

    @task.bash
    def register_model():
        return f"{STOA} src.models.flow.registry --promote"

    (
        train_candidates()
        >> select_model()
        >> mitigate_bias()
        >> check_gates()
        >> register_model()
    )


stoa_train()
