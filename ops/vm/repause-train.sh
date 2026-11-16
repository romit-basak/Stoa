#!/usr/bin/env bash
# Pause stoa_train again once none of its runs is queued or running (started by train.sh as a
# transient systemd unit, so it outlives the SSH session). See train.sh for why it stays paused.
set -uo pipefail
cd /srv/stoa
sleep 120
while true; do
  n=$(docker compose exec -T postgres psql -U airflow -d airflow -tAc \
    "SELECT count(*) FROM dag_run WHERE dag_id = 'stoa_train' AND state IN ('queued', 'running')")
  # Anything but 0 (a number or an error while the stack restarts) means wait.
  [ "$n" = 0 ] && break
  sleep 300
done
docker compose exec -T airflow-scheduler airflow dags pause stoa_train
logger -t stoa-train "stoa_train run over; paused it again"
