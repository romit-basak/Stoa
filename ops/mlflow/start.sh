# Start the MLflow tracking server (docker-compose service `mlflow`).
#
# Runs, parameters, metrics and the model registry live in a `mlflow` database on the compose
# Postgres, created here on the first start (the Postgres init scripts only run on an empty
# volume, so they would miss every existing Airflow install). Model files and other artifacts
# go to MLFLOW_ARTIFACTS_DESTINATION; --serve-artifacts makes the server upload and download
# them, so clients never need bucket credentials.
set -eu

python - <<'PY'
import os
import time

import psycopg2

dsn = os.environ["MLFLOW_ADMIN_DSN"]
for attempt in range(30):
    try:
        conn = psycopg2.connect(dsn)
        break
    except psycopg2.OperationalError:
        time.sleep(2)
else:
    raise SystemExit("Postgres did not come up")
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'mlflow'")
    if cur.fetchone() is None:
        cur.execute("CREATE DATABASE mlflow")
        print("Created the mlflow database")
conn.close()
PY

# Host headers the server accepts: the browser on the host (localhost:5001) and the Airflow
# containers (mlflow:5000). Anything else is refused (DNS-rebinding protection).
exec mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2 \
    --allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,mlflow,mlflow:*"
