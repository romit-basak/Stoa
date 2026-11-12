#!/usr/bin/env bash
# Run Airflow and MLflow on a Compute Engine VM in the team project (docs/pipeline/cloud_vm.md).
#
# Runs on a teammate's laptop with gcloud logged in. The VM has no public IP: every connection
# goes through Identity-Aware Proxy (IAP), and the UIs are reached through an SSH tunnel.
#
#   ops/vm/vm.sh tunnel       # start the VM if needed, forward localhost:8080 (Airflow) and :5001 (MLflow)
#   ops/vm/vm.sh start|stop|status|ssh
#   ops/vm/vm.sh deploy       # pull main on the VM and restart the stack (refuses during a run)
#   ops/vm/vm.sh resize e2-highmem-4   # change the machine type (stops the VM first)
#
# One-time setup (done once for the team, not per person):
#   ops/vm/vm.sh create       # firewall rule, NAT, the VM itself
#   ops/vm/vm.sh setup        # install Docker, clone the repo, copy .env and the bucket key, start
#   ops/vm/vm.sh schedule     # start the VM at 23:40 UTC each day for the 00:00 UTC pipeline run
#
# The VM stops itself once nothing needs it (ops/vm/idle-stop.sh), so it costs money only while
# a DAG run, an SSH session or a tunnel is active.
set -euo pipefail

PROJECT=${STOA_GCP_PROJECT:-stoa-511822}
ZONE=${STOA_VM_ZONE:-us-east1-b} # same region as gs://stoa-dvc-data, so bucket reads are free
REGION=${ZONE%-*}
VM=${STOA_VM_NAME:-stoa-ops}
MACHINE=${STOA_VM_MACHINE:-e2-standard-4} # 4 vCPU, 16 GB
DISK_GB=${STOA_VM_DISK_GB:-50}
TAG=stoa-ops
SCHEDULE=stoa-ops-daily-start
IAP_RANGE=35.235.240.0/20 # where IAP connections come from
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

g() { gcloud --project "$PROJECT" "$@"; }
vm_ssh() { g compute ssh "$VM" --zone "$ZONE" --tunnel-through-iap "$@"; }
vm_state() { g compute instances describe "$VM" --zone "$ZONE" --format='value(status)'; }

start() {
  if [ "$(vm_state)" != RUNNING ]; then
    echo "Starting $VM..."
    g compute instances start "$VM" --zone "$ZONE" --quiet
  fi
  # sshd comes up a little after the instance reports RUNNING.
  for _ in $(seq 30); do
    vm_ssh --command true -- -o ConnectTimeout=5 >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "VM is running but SSH isn't answering yet; try again in a minute." >&2
  return 1
}

create() {
  g services enable iap.googleapis.com compute.googleapis.com

  if ! g compute firewall-rules describe stoa-allow-iap-ssh >/dev/null 2>&1; then
    g compute firewall-rules create stoa-allow-iap-ssh --network default --direction INGRESS \
      --allow tcp:22 --source-ranges "$IAP_RANGE" --target-tags "$TAG" \
      --description "SSH to the Stoa ops VM through IAP only"
  fi

  # No public IP: outbound traffic (Docker Hub, the Melbourne and weather APIs) goes through
  # Cloud NAT, and Google APIs (the bucket) through Private Google Access.
  if ! g compute routers describe stoa-router --region "$REGION" >/dev/null 2>&1; then
    g compute routers create stoa-router --network default --region "$REGION"
  fi
  if ! g compute routers nats describe stoa-nat --router stoa-router --region "$REGION" >/dev/null 2>&1; then
    g compute routers nats create stoa-nat --router stoa-router --region "$REGION" \
      --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
  fi
  g compute networks subnets update default --region "$REGION" --enable-private-ip-google-access

  # No service account on the VM itself: the containers use the stoa-airflow-dvc key, as on
  # laptops, so the VM can't do anything else in the project.
  g compute instances create "$VM" --zone "$ZONE" --machine-type "$MACHINE" \
    --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
    --no-address --tags "$TAG" --no-service-account --no-scopes \
    --shielded-secure-boot --metadata enable-oslogin=TRUE
}

setup() {
  [ -f "$ROOT/.env" ] || { echo "No .env in $ROOT" >&2; exit 1; }
  [ -f "$ROOT/.secrets/stoa-airflow-dvc.json" ] || { echo "No bucket key in .secrets/" >&2; exit 1; }
  start
  vm_ssh --command 'rm -rf ~/stoa-upload && mkdir -m 700 ~/stoa-upload'
  g compute scp --zone "$ZONE" --tunnel-through-iap \
    "$ROOT/ops/vm/setup.sh" "$ROOT/.env" "$ROOT/.secrets/stoa-airflow-dvc.json" \
    "$VM:~/stoa-upload/"
  vm_ssh --command "sudo bash ~/stoa-upload/setup.sh ${STOA_VM_BRANCH:-main} && rm -rf ~/stoa-upload"
}

schedule() {
  # Compute Engine starts the VM itself; its service agent needs permission to do so
  # (granted once by a project owner, see docs/pipeline/cloud_vm.md).
  if ! g compute resource-policies describe "$SCHEDULE" --region "$REGION" >/dev/null 2>&1; then
    g compute resource-policies create instance-schedule "$SCHEDULE" --region "$REGION" \
      --vm-start-schedule "40 23 * * *" --timezone UTC \
      --description "Start the Stoa ops VM before the 00:00 UTC stoa_pipeline run"
  fi
  g compute instances add-resource-policies "$VM" --zone "$ZONE" --resource-policies "$SCHEDULE"
}

case "${1:-}" in
  create) create ;;
  setup) setup ;;
  schedule) schedule ;;
  start) start && echo "$VM is up. ops/vm/vm.sh tunnel to open the UIs." ;;
  stop) g compute instances stop "$VM" --zone "$ZONE" --quiet ;;
  status)
    echo "$VM: $(vm_state)"
    if [ "$(vm_state)" = RUNNING ]; then
      vm_ssh --command 'cd /srv/stoa && sudo docker compose ps --format "{{.Service}}\t{{.Status}}"; journalctl -t stoa-idle -n 3 --no-pager -q'
    fi
    ;;
  ssh) start && vm_ssh ;;
  tunnel)
    start
    echo "Airflow: http://localhost:8080   MLflow: http://localhost:5001   (Ctrl-C to close)"
    echo "The VM stays up while this tunnel is open and stops itself about 15 minutes after."
    vm_ssh -- -N -L 8080:localhost:8080 -L 5001:localhost:5001
    ;;
  deploy) start && vm_ssh --command 'sudo bash /srv/stoa/ops/vm/deploy.sh' ;;
  resize)
    [ -n "${2:-}" ] || { echo "usage: $0 resize <machine-type>" >&2; exit 1; }
    g compute instances stop "$VM" --zone "$ZONE" --quiet
    g compute instances set-machine-type "$VM" --zone "$ZONE" --machine-type "$2"
    start
    ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
