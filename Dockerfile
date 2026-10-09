# Airflow image with Stoa's Python dependencies. The repo itself is mounted at /opt/stoa
# (see docker-compose.yaml), so code changes don't need a rebuild; dependency changes do.
FROM apache/airflow:3.3.2-python3.11

# Read the dependency list straight from pyproject.toml so it is never duplicated here.
# Pinning apache-airflow stops pip from upgrading or downgrading Airflow itself.
COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
