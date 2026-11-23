#!/usr/bin/env bash
# Deploy the 15-minute Melbourne minute-feed poller to GCP Cloud Run Jobs + Cloud Scheduler.
#
# Replaces the off-GCP Explorer HPC cron (ops/hpc/) so the entire STOA pipeline runs on GCP.
# Writes directly to gs://stoa-dvc-data/minute_feed (configured in configs/data.yaml).
#
# Usage:
#   ops/gcp/deploy_minute_feed_job.sh
set -euo pipefail

PROJECT=${STOA_GCP_PROJECT:-stoa-511822}
REGION=${STOA_GCP_REGION:-us-east1}
REPO_NAME=${STOA_AR_REPO:-stoa}
JOB=${STOA_MINUTE_JOB:-stoa-minute-feed}
SCHEDULER_NAME=${STOA_MINUTE_SCHEDULER:-stoa-minute-feed-15m}
SA_EMAIL=${STOA_MINUTE_SA:-stoa-airflow-dvc@${PROJECT}.iam.gserviceaccount.com}
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO_NAME}/${JOB}:latest"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

gcloud --project "$PROJECT" services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com

if ! gcloud --project "$PROJECT" artifacts repositories describe "$REPO_NAME" \
  --location "$REGION" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --description="STOA container images"
fi

cp "$ROOT/ops/gcp/minute_feed.Dockerfile" "$ROOT/Dockerfile.minutefeed.tmp"
trap 'rm -f "$ROOT/Dockerfile.minutefeed.tmp"' EXIT

gcloud --project "$PROJECT" builds submit "$ROOT" \
  --config /dev/stdin <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'Dockerfile.minutefeed.tmp', '-t', '${IMAGE}', '.']
images:
  - '${IMAGE}'
EOF

gcloud --project "$PROJECT" run jobs deploy "$JOB" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --memory 1Gi \
  --cpu 1 \
  --task-timeout 10m \
  --max-retries 2

JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${JOB}:run"
if gcloud --project "$PROJECT" scheduler jobs describe "$SCHEDULER_NAME" \
  --location "$REGION" >/dev/null 2>&1; then
  gcloud --project "$PROJECT" scheduler jobs update http "$SCHEDULER_NAME" \
    --location "$REGION" \
    --schedule "*/15 * * * *" \
    --uri "$JOB_URI" \
    --http-method POST \
    --oauth-service-account-email "$SA_EMAIL"
else
  gcloud --project "$PROJECT" scheduler jobs create http "$SCHEDULER_NAME" \
    --location "$REGION" \
    --schedule "*/15 * * * *" \
    --uri "$JOB_URI" \
    --http-method POST \
    --oauth-service-account-email "$SA_EMAIL" \
    --description "Poll Melbourne minute pedestrian feed into gs://stoa-dvc-data/minute_feed every 15 minutes"
fi

echo "Deployed Cloud Run Job $JOB and Cloud Scheduler $SCHEDULER_NAME (every 15 min) in $PROJECT ($REGION)."
