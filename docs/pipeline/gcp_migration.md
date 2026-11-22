# Complete GCP Architecture & Migration Guide

STOA runs end-to-end in GCP project **`stoa-511822`** (`us-east1`), combining the cost-optimized on-demand Compute Engine VM (`stoa-ops`, ~$10/month) with serverless Cloud Run and Cloud Storage for 24/7 components.

## Architecture Overview

| Component | GCP Service | Details |
|---|---|---|
| **Data & DVC remote** | Cloud Storage | `gs://stoa-dvc-data` (versioned data, `mlflow-artifacts/`, `minute_feed/`, `backups/postgres/`, `dashboard/`) |
| **Daily pipeline, training & site builds** | Compute Engine (`stoa-ops`) | `e2-standard-4` in `us-east1-b` running Docker Compose (Airflow 3.3.2 + `ffmpeg`/`imageio`, Postgres 16, MLflow, STOA Dashboard). Starts daily at `23:40 UTC` and auto-stops when idle. |
| **Always-on STOA Dashboard** | Cloud Run (`stoa-dashboard`) | Stateless container (`ops/gcp/dashboard.Dockerfile`) syncing `gs://stoa-dvc-data/dashboard` published automatically by `build_dashboard` and `site_build_dashboard`. |
| **15-min Minute-Feed Poller** | Cloud Run Job + Cloud Scheduler | `stoa-minute-feed` triggered every 15 minutes (`*/15 * * * *`) writing directly to `gs://stoa-dvc-data/minute_feed`, replacing the Explorer HPC cron (`ops/hpc/`). |
| **CI/CD Retraining** | GitHub Actions → IAP SSH | `.github/workflows/train.yml` triggers `ops/vm/vm.sh train` on `stoa-ops`. |

## Deployment Steps

Ensure `gcloud` is authenticated with your team account (`gcloud auth login && gcloud config set project stoa-511822`):

1. **Deploy the updated Airflow + Simulation + Dashboard stack to `stoa-ops`:**
   ```bash
   ops/vm/vm.sh deploy
   ```
   This pulls `main`, rebuilds the Airflow image with `ffmpeg` + `imageio`, binds the dashboard to `127.0.0.1:8501`, and sets `STOA_DASHBOARD_GCS=gs://stoa-dvc-data/dashboard` on `airflow-scheduler`.

2. **Open the IAP tunnel (Airflow `:8080`, MLflow `:5001`, Dashboard `:8501`):**
   ```bash
   ops/vm/vm.sh tunnel
   ```

3. **Deploy the 24/7 Cloud Run STOA Dashboard:**
   ```bash
   chmod +x ops/gcp/deploy_dashboard_cloudrun.sh
   ops/gcp/deploy_dashboard_cloudrun.sh
   ```
   *(Pass `STOA_DASHBOARD_PUBLIC=1` if you want unauthenticated public web access; omit it for IAM-protected team access.)*

4. **Migrate the 15-minute Minute-Feed Poller to Cloud Run Jobs + Cloud Scheduler:**
   ```bash
   chmod +x ops/gcp/deploy_minute_feed_job.sh
   ops/gcp/deploy_minute_feed_job.sh
   ```
   Once verified (`gcloud run jobs executions list --job stoa-minute-feed --region us-east1`), you can remove the old HPC cron in `ops/hpc/`.
