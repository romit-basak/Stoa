#!/usr/bin/env bash
# Back up the ops VM's Postgres to the bucket (run as root: stoa-backup.timer, or `vm.sh backup`).
#
# The Postgres volume holds everything on the VM that exists nowhere else: MLflow's runs and
# model registry (database `mlflow`; the model files are already in gs://stoa-dvc-data/
# mlflow-artifacts) and Airflow's run history (database `airflow`). Each database is dumped
# with pg_dump (custom format, compressed), checked, and uploaded to
# gs://stoa-dvc-data/backups/postgres/<UTC stamp>/, with a COMPLETE marker written last.
# Backups older than STOA_BACKUP_KEEP_DAYS (default 30) are pruned; the newest is always kept.
#
# pg_dump reads a consistent snapshot, so it is safe while DAG runs are writing.
set -euo pipefail
cd /srv/stoa

KEEP_DAYS=${STOA_BACKUP_KEEP_DAYS:-30}
DATABASES=(airflow mlflow)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
GCS_PY=$(cat ops/vm/backup_gcs.py)
log() { echo "$*"; logger -t stoa-backup "$*"; }
# The mlflow image has google-cloud-storage and the bucket key; `run` works whether or not the
# MLflow server is up.
gcs() {
  docker compose run --rm --no-deps -T -e STOA_BACKUP_BUCKET -e STOA_BACKUP_PREFIX mlflow \
    python -c "$GCS_PY" "$@"
}

TMP=$(mktemp -d /var/tmp/stoa-backup.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

# At boot (the timer fires at 23:50 UTC, 10 minutes after the scheduled start) Postgres may
# still be starting.
for _ in $(seq 60); do
  docker compose exec -T postgres pg_isready -U airflow -q && break
  sleep 5
done

for db in "${DATABASES[@]}"; do
  docker compose exec -T postgres pg_dump -U airflow -Fc -d "$db" > "$TMP/$db.dump"
  # A dump pg_restore can't read is worse than none: it would look like a backup.
  docker compose exec -T postgres pg_restore --list < "$TMP/$db.dump" > /dev/null
done

for db in "${DATABASES[@]}"; do
  gcs put "$STAMP/$db.dump" < "$TMP/$db.dump"
done
printf 'airflow mlflow\n' | gcs put "$STAMP/COMPLETE"

sizes=$(cd "$TMP" && du -h ./*.dump | awk '{sub("./", "", $2); printf "%s %s, ", $2, $1}')
log "backed up to gs://${STOA_BACKUP_BUCKET:-stoa-dvc-data}/${STOA_BACKUP_PREFIX:-backups/postgres}/$STAMP/ (${sizes%, })"

gcs prune "$KEEP_DAYS" || log "pruning old backups failed (the new backup is fine)"
