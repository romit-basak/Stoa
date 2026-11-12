#!/usr/bin/env bash
# Set up the Stoa ops VM (run as root on the VM by `ops/vm/vm.sh setup`; safe to re-run).
#
# Expects .env and stoa-airflow-dvc.json next to this script (vm.sh copies them). Installs Docker,
# clones the repo to /srv/stoa as the `stoa` user (uid 50000, the containers' user), pulls the
# committed data version, installs the boot and idle-stop units, and starts the stack.
# DAGs stay paused: unpausing stoa_pipeline starts a run that emails the team and pushes data.
set -euo pipefail

BRANCH=${1:-main}
REPO=https://github.com/romit-basak/Stoa.git
DIR=/srv/stoa
UPLOAD=$(cd "$(dirname "$0")" && pwd)
as_stoa() { sudo -u stoa "$@"; }

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
apt-get install -y -q git >/dev/null

id stoa >/dev/null 2>&1 || useradd --system --uid 50000 --gid 0 --home-dir "$DIR" \
  --shell /usr/sbin/nologin stoa
if [ ! -d "$DIR/.git" ]; then
  install -d -o stoa -g 0 "$DIR"
  as_stoa git clone --branch "$BRANCH" "$REPO" "$DIR"
fi

# Secrets: the laptop's .env plus the VM's own settings, and the bucket key.
install -d -o stoa -g 0 -m 700 "$DIR/.secrets"
install -o stoa -g 0 -m 600 "$UPLOAD/stoa-airflow-dvc.json" "$DIR/.secrets/"
{
  sed '/^# --- VM settings/,$d' "$UPLOAD/.env"
  cat <<'EOF'
# --- VM settings (ops/vm/setup.sh rewrites everything below this line) ---
COMPOSE_FILE=docker-compose.yaml:ops/vm/compose.vm.yaml
AIRFLOW_UID=50000
STOA_MLFLOW_ARTIFACTS=gs://stoa-dvc-data/mlflow-artifacts
EOF
} > "$DIR/.env.new"
install -o stoa -g 0 -m 600 "$DIR/.env.new" "$DIR/.env" && rm "$DIR/.env.new"

cd "$DIR"
docker compose build
# The committed data version, so the first run is incremental like on a laptop.
docker compose run --rm --no-deps -e DVC_NO_ANALYTICS=1 airflow-scheduler bash -c 'cd /opt/stoa && dvc pull'

install -m 644 ops/vm/stoa-compose.service ops/vm/stoa-idle-stop.service \
  ops/vm/stoa-idle-stop.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now stoa-compose.service stoa-idle-stop.timer

echo "Stack started. DAGs are paused; see docs/pipeline/cloud_vm.md before unpausing."
