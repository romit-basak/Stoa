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
# dvc_push leaves data/*.dvc and dvc.lock changed but uncommitted (as on laptops); those versions
# are already in the bucket. Set them aside for the pull, then put them back: they describe the
# data on this disk. If main changed them too, main's copies win and the VM's stay in a stash;
# the next dvc_push records the data on disk again either way.
POINTERS=(data/raw.dvc data/interim.dvc data/processed.dvc dvc.lock)
stashed=""
if [ -n "$(sudo -u stoa git status --porcelain -- "${POINTERS[@]}")" ]; then
  sudo -u stoa git stash push -q -m "vm dvc pointers, deploy $(date -u +%FT%TZ)" -- "${POINTERS[@]}"
  stashed=1
fi
if ! sudo -u stoa git pull --ff-only; then
  # Leave nothing half done: restart the scheduler on the old code and give the pointers back.
  [ -n "$stashed" ] && sudo -u stoa git stash pop -q
  docker compose start airflow-scheduler
  exit 1
fi
if [ -n "$stashed" ] && ! sudo -u stoa git stash pop -q 2>/dev/null; then
  sudo -u stoa git checkout -q HEAD -- "${POINTERS[@]}"
  echo "main changed the DVC pointers too: kept main's; the VM's are in git stash" >&2
fi
docker compose up -d --build
# The backup timer came after the VM was set up; installing it here ships unit changes with main.
install -m 644 ops/vm/stoa-backup.service ops/vm/stoa-backup.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable -q --now stoa-backup.timer
sudo -u stoa git log --oneline -1
