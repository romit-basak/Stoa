# Lightweight 24/7 Cloud Run image for the STOA web dashboard.
# Pulls the latest built bundle (index.html, dashboard_data.json, *.mp4) from GCS on startup
# and serves it on $PORT so the dashboard stays live even while the stoa-ops VM is asleep.
FROM python:3.11-slim

WORKDIR /opt/stoa
RUN pip install --no-cache-dir google-cloud-storage>=2.18

COPY src/dashboard/__init__.py src/dashboard/serve.py src/dashboard/

ENV PORT=8080
ENV STOA_DASHBOARD_GCS=gs://stoa-dvc-data/dashboard

CMD ["python", "-m", "src.dashboard.serve", "--dir", "/tmp/stoa-dashboard", "--host", "0.0.0.0"]
