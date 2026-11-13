# Airflow image with Stoa's Python dependencies. The repo itself is mounted at /opt/stoa
# (see docker-compose.yaml), so code changes don't need a rebuild; dependency changes do.
FROM apache/airflow:3.3.2-python3.11

# LightGBM's compiled library links against OpenMP (libgomp1), which the Airflow base image
# doesn't ship; without it `import lightgbm` fails and stoa_train crashes at the lgbm candidate.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Read the dependency list straight from pyproject.toml so it is never duplicated here.
# Pinning apache-airflow stops pip from upgrading or downgrading Airflow itself.
COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
