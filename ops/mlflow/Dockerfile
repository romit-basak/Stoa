# MLflow tracking server for Stoa (docker-compose service `mlflow`).
# Run records go to a `mlflow` database on the compose Postgres; model files go to
# STOA_MLFLOW_ARTIFACTS (a local volume by default, or gs://... on a team machine or the VM).
FROM python:3.11-slim

RUN pip install --no-cache-dir \
        mlflow==3.17.2 \
        psycopg2-binary==2.9.13 \
        google-cloud-storage==3.17.1

COPY start.sh /start.sh
EXPOSE 5000
CMD ["sh", "/start.sh"]
