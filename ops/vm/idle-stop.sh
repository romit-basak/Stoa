#!/usr/bin/env bash
# Power the ops VM off once nothing needs it (systemd timer stoa-idle-stop, every 5 minutes).
#
# The VM counts as busy while it has been up under 30 minutes, an Airflow DAG run is queued or
# running, or anyone is connected over SSH (which includes `vm.sh tunnel`). After 3 idle checks
# in a row (about 15 minutes) it shuts down; a stopped VM costs only its disk.
set -u

MIN_UPTIME=${STOA_IDLE_MIN_UPTIME:-1800}
CHECKS=${STOA_IDLE_CHECKS:-3}
COUNT_FILE=/run/stoa-idle-count
log() { logger -t stoa-idle "$*"; }

uptime_s=$(cut -d. -f1 /proc/uptime)
ssh_sessions=$(ss -Htn state established '( sport = :22 )' | wc -l)
runs=$(cd /srv/stoa && docker compose exec -T postgres psql -U airflow -d airflow -tAc \
  "SELECT count(*) FROM dag_run WHERE state IN ('queued', 'running')" 2>/dev/null) || runs=unknown

busy=""
if [ "$uptime_s" -lt "$MIN_UPTIME" ]; then
  busy="up ${uptime_s}s"
elif [ "$ssh_sessions" -gt 0 ]; then
  busy="$ssh_sessions SSH session(s)"
elif [ "$runs" = unknown ] && [ "$uptime_s" -lt 7200 ]; then
  # Postgres not answering (still starting, or the stack is down): wait, but not forever.
  busy="Airflow database not answering"
elif [ "$runs" != unknown ] && [ "$runs" -gt 0 ]; then
  busy="$runs DAG run(s) active"
fi

if [ -n "$busy" ]; then
  echo 0 > "$COUNT_FILE"
  exit 0
fi

count=$(( $(cat "$COUNT_FILE" 2>/dev/null || echo 0) + 1 ))
echo "$count" > "$COUNT_FILE"
log "idle check $count of $CHECKS"
if [ "$count" -ge "$CHECKS" ]; then
  log "idle for $CHECKS checks: shutting down"
  systemctl poweroff
fi
