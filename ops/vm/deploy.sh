#!/usr/bin/env bash
# Update the ops VM to the latest main and restart the stack (run as root by `vm.sh deploy`).
#
# Refuses while a DAG run is active: tasks would otherwise run half old, half new code. The
# scheduler is stopped during the pull so no new task starts in between.
set -euo pipefail
cd /srv/stoa

runs=$(docker compose exec -T postgres psql -U airflow -d airflow -tAc \
  "SELECT count(*) FROM dag_run WHERE state IN ('queued', 'running')")
if [ "$runs" -gt 0 ]; then
  echo "$runs DAG run(s) active; deploy again once they finish." >&2
  exit 1
fi

docker compose stop airflow-scheduler
# dvc_push leaves data/*.dvc and dvc.lock changed but uncommitted (as on laptops). If main changed
# them too, the pull stops here and the scheduler stays stopped until someone sorts it out.
sudo -u stoa git pull --ff-only
docker compose up -d --build
sudo -u stoa git log --oneline -1
