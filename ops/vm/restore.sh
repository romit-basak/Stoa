#!/usr/bin/env bash
# Restore the ops VM's Postgres from a bucket backup (run as root by `vm.sh restore`).
#
#   restore.sh <stamp>|latest
#
# Replaces the airflow and mlflow databases with the backup's. Refuses during a DAG run. Takes a
# fresh backup of the current state first, so a restore can itself be undone. Airflow and MLflow
# are stopped while the databases are swapped; `docker compose up` then migrates them if the
# backup predates the current Airflow version.
set -euo pipefail
cd /srv/stoa

WANT=${1:?usage: restore.sh <stamp>|latest}
DATABASES=(airflow mlflow)
APPS=(airflow-scheduler airflow-dag-processor airflow-apiserver mlflow)
GCS_PY=$(cat ops/vm/backup_gcs.py)
gcs() {
  docker compose run --rm --no-deps -T -e STOA_BACKUP_BUCKET -e STOA_BACKUP_PREFIX mlflow \
    python -c "$GCS_PY" "$@"
}
psql_admin() { docker compose exec -T postgres psql -U airflow -d postgres -v ON_ERROR_STOP=1 -qc "$1"; }

stamps=$(gcs list)
if [ "$WANT" = latest ]; then
  STAMP=$(head -n1 <<<"$stamps")
else
  STAMP=$(grep -x -- "$WANT" <<<"$stamps" || true)
fi
[ -n "$STAMP" ] || { echo "No complete backup '$WANT'. Available:" >&2; echo "$stamps" >&2; exit 1; }

runs=$(docker compose exec -T postgres psql -U airflow -d airflow -tAc \
  "SELECT count(*) FROM dag_run WHERE state IN ('queued', 'running')" 2>/dev/null || echo 0)
if [ "$runs" -gt 0 ]; then
  echo "$runs DAG run(s) active; restore once they finish." >&2
  exit 1
fi

TMP=$(mktemp -d /var/tmp/stoa-restore.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
for db in "${DATABASES[@]}"; do
  gcs get "$STAMP/$db.dump" > "$TMP/$db.dump"
  docker compose exec -T postgres pg_restore --list < "$TMP/$db.dump" > /dev/null
done

echo "Backing up the current state first..."
bash ops/vm/backup.sh

echo "Restoring $STAMP..."
docker compose stop "${APPS[@]}"
for db in "${DATABASES[@]}"; do
  psql_admin "DROP DATABASE IF EXISTS $db WITH (FORCE)"
  psql_admin "CREATE DATABASE $db"
  docker compose exec -T postgres pg_restore -U airflow -d "$db" --no-owner --exit-on-error \
    < "$TMP/$db.dump"
done
docker compose up -d
logger -t stoa-backup "restored $STAMP"
echo "Restored $STAMP. DAGs keep the paused state they had in the backup."
