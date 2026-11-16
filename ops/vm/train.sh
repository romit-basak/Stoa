#!/usr/bin/env bash
# Deploy main and start stoa_train on the ops VM (run as root by `vm.sh train`, which the GitHub
# workflow .github/workflows/train.yml calls on a push to model code and once a month).
#
#   sudo bash /srv/stoa/ops/vm/train.sh [--tune]   (vm.sh train runs main's copy of it)
#
# 1. Waits for active DAG runs to finish (up to STOA_TRAIN_WAIT minutes, default 90): deploying
#    mid-run would mix old and new code.
# 2. Deploys main (deploy.sh), so the run trains the code that was just pushed.
# 3. Unpauses stoa_train and triggers a run, then re-pauses it once no stoa_train run is left.
#    stoa_train stays paused between runs: its @monthly slot falls at 00:00 UTC, when the VM is
#    off or busy with stoa_pipeline, and unpausing on a later start would train alongside it.
#    If unpausing creates the missed monthly run by itself, that run is the training run and no
#    second one is triggered.
#
# Returns as soon as the run is queued; the run emails the team when it finishes (notify_team)
# or fails. The VM stays up while the run is active (idle-stop.sh) and stops itself after.
set -euo pipefail
cd /srv/stoa

TUNE=false
[ "${1:-}" = --tune ] && TUNE=true
WAIT_MIN=${STOA_TRAIN_WAIT:-90}

psql_q() { docker compose exec -T postgres psql -U airflow -d airflow -tAc "$1"; }
airflow() { docker compose exec -T airflow-scheduler airflow "$@"; }
active() { psql_q "SELECT count(*) FROM dag_run WHERE state IN ('queued', 'running')${1:-}"; }

for i in $(seq "$WAIT_MIN"); do
  [ "$(active)" -eq 0 ] && break
  [ "$i" -eq "$WAIT_MIN" ] && { echo "DAG runs still active after $WAIT_MIN min" >&2; exit 1; }
  [ "$i" -eq 1 ] && echo "waiting for active DAG runs to finish..."
  sleep 60
done

# deploy.sh as it is on main, so a fix to it applies to this deploy too.
sudo -u stoa git fetch -q origin main
bash <(sudo -u stoa git show origin/main:ops/vm/deploy.sh)
# The scheduler and dag-processor need a moment to parse the new code after the restart.
for _ in $(seq 30); do
  airflow dags details stoa_train >/dev/null 2>&1 && break
  sleep 10
done

airflow dags unpause stoa_train >/dev/null
sleep 30 # let the scheduler create the missed monthly run, if there is one
if [ "$(active " AND dag_id = 'stoa_train'")" -gt 0 ]; then
  echo "unpausing started the scheduled monthly run (tune=false); not triggering another"
else
  airflow dags trigger stoa_train --conf "{\"tune\": $TUNE}" \
    --run-id "ci__$(date -u +%Y%m%dT%H%M%S)" >/dev/null
  echo "triggered stoa_train (tune=$TUNE)"
fi

# Re-pause once the training is over, in a unit of its own so it outlives this SSH session.
systemd-run --quiet --collect --unit "stoa-train-repause-$(date +%s)" \
  bash /srv/stoa/ops/vm/repause-train.sh
echo "stoa_train will be paused again when the run ends"
