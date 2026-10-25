#!/bin/bash
# One poll of the per-minute counts into the project bucket, run by Slurm's scrontab on
# Northeastern's Explorer cluster (ops/hpc/README.md). Same code as the stoa_minute_feed DAG.
set -euo pipefail
cd "$HOME/stoa"
module load python/3.13.5
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/stoa/minute-feed-sa.json"
exec .venv-feed/bin/python -m src.ingest.minute_feed --bucket
