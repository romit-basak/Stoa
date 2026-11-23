#!/usr/bin/env bash
# Build and deploy the always-on STOA monitoring & simulation dashboard to Cloud Run.
#
# Runs in the team GCP project (stoa-511822, us-east1). Syncs built dashboard assets from
# gs://stoa-dvc-data/dashboard (published automatically by Airflow's build_dashboard task),
# so the dashboard remains accessible 24/7 even when the stoa-ops VM is stopped.
#
# Usage:
#   ops/gcp/deploy_dashboard_cloudrun.sh
#   STOA_DASHBOARD_PUBLIC=1 ops/gcp/deploy_dashboard_cloudrun.sh   # allow unauthenticated access
set -euo pipefail

PROJECT=${STOA_GCP_PROJECT:-stoa-511822}
REGION=${STOA_GCP_REGION:-us-east1}
REPO_NAME=${STOA_AR_REPO:-stoa}
SERVICE=${STOA_DASHBOARD_SERVICE:-stoa-dashboard}
SA_EMAIL=${STOA_DASHBOARD_SA:-stoa-airflow-dvc@${PROJECT}.iam.gserviceaccount.com}
GCS_URI=${STOA_DASHBOARD_GCS:-gs://stoa-dvc-data/dashboard}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO_NAME}/${SERVICE}:latest"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

gcloud --project "$PROJECT" services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

if ! gcloud --project "$PROJECT" artifacts repositories describe "$REPO_NAME" \
  --location "$REGION" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --description="STOA container images"
fi

# Build from repo root using ops/gcp/dashboard.Dockerfile
cp "$ROOT/ops/gcp/dashboard.Dockerfile" "$ROOT/Dockerfile.cloudrun.tmp"
trap 'rm -f "$ROOT/Dockerfile.cloudrun.tmp"' EXIT

gcloud --project "$PROJECT" builds submit "$ROOT" \
  --config /dev/stdin <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'Dockerfile.cloudrun.tmp', '-t', '${IMAGE}', '.']
images:
  - '${IMAGE}'
EOF

AUTH_FLAG="--no-allow-unauthenticated"
if [ "${STOA_DASHBOARD_PUBLIC:-0}" = "1" ]; then
  AUTH_FLAG="--allow-unauthenticated"
fi

gcloud --project "$PROJECT" run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --set-env-vars "STOA_DASHBOARD_GCS=${GCS_URI}" \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  "$AUTH_FLAG"

echo "Deployed $SERVICE in $PROJECT ($REGION)."
echo "Trigger a live reload after a new pipeline run via: <service-url>/_refresh"
