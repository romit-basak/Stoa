# Lightweight Cloud Run Job image for the 15-minute Melbourne minute-feed poller.
# Replaces the Explorer HPC cron (ops/hpc/) so 100% of the STOA pipeline runs on GCP.
FROM python:3.11-slim

WORKDIR /opt/stoa
RUN pip install --no-cache-dir \
    google-cloud-storage>=2.18 \
    pandas>=2.2 \
    pyarrow>=15 \
    pyyaml>=6 \
    requests>=2.31

COPY configs/data.yaml configs/data.yaml
COPY src/__init__.py src/__init__.py
COPY src/ingest/__init__.py src/ingest/common.py src/ingest/minute_feed.py src/ingest/
COPY src/pipeline/__init__.py src/pipeline/dvc_push.py src/pipeline/

CMD ["python", "-m", "src.ingest.minute_feed", "--bucket"]
