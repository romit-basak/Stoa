# Experiment tracking (MLflow)

Every model run is logged to one MLflow server: its settings, metrics, the model file, and the code and data it was trained on. Comparing candidates, picking the best and later pushing it to the registry all start from these runs.

## Start the server

MLflow is a service in `docker-compose.yaml`, next to Airflow:

```bash
docker compose up -d --build mlflow     # or `docker compose up -d --build` for everything
open http://localhost:5001              # runs, comparisons and the model registry
```

It listens on 5001 on your machine (macOS uses 5000 for AirPlay) and on `mlflow:5000` inside the compose network, which is where Airflow tasks log. Runs are stored in a `mlflow` database on the compose Postgres, created on the first start, so they survive `docker compose down` (but not `down -v`). Check the setup end to end with:

```bash
.venv/bin/python -m src.models.tracking --smoke     # logs and reloads a tiny model (experiment smoke-test)
```

## Log a training run

Wrap the training in `tracking.run` and log metrics and the model inside it:

```python
import mlflow
from src.models import tracking

params = {"loss": "poisson", "max_iter": 300, "learning_rate": 0.1}
with tracking.run("hgb_poisson", params=params):
    model = HistGradientBoostingRegressor(**params).fit(x_train, y_train)
    mlflow.log_metrics({"wape_cv": 0.41, "mae_cv": 118.0, "rmse_cv": 240.0})
    tracking.log_sklearn_model(model, input_example=x_train[:5])
```

`tracking.run` adds these on its own, so every run can be traced back to its inputs:

| Tag or artifact | What it records |
|---|---|
| `stoa.git_commit`, `stoa.git_dirty` | The commit the code ran from, and whether it had uncommitted changes (`true` means the run can't be reproduced exactly from the commit) |
| `stoa.dvc.raw`, `stoa.dvc.interim`, `stoa.dvc.processed` | The DVC version (md5) of each data folder. Check out the commit and run `dvc checkout` to get that data back |
| `data/frame_spec.json` and `stoa.frame.*` | The training frame's spec: feature lists, input hashes, OSM extract, row and sensor counts (`docs/features/training_frame.md`). `stoa.frame_spec = missing` means the frame wasn't built; run `python -m src.features.training_frame` first |

`log_sklearn_model` saves in skops, MLflow's safe format, and lists the model's own scikit-learn types as trusted, so `mlflow.sklearn.load_model(uri)` works with no extra arguments. LightGBM and XGBoost have their own `mlflow.lightgbm` / `mlflow.xgboost` loggers.

## Conventions, so runs can be compared

- **One experiment for the flow model:** `stoa-flow` (the default, `configs/models.yaml`). Use another only for scratch work (for example `scratch-<name>`).
- **Run name** = the model family and variant, for example `hgb_poisson`, `lgbm_portable` or `baseline_hour_of_week`.
- **Metric names:** `wape_cv`, `mae_cv` and `rmse_cv` for the mean over the 5 location folds, and `wape_fold0` to `wape_fold4` for each fold. `wape_holdout` (the held-out precincts) and `wape_nyc` are for the final candidate only (CLAUDE.md core rule 2: choices are made on the folds, never on the holdouts).
- **Params:** every setting that changes the model, including the feature variant (`portable` or `rich`) and the training-data window.
- Commit before a run you want to keep, so `stoa.git_dirty` is `false`.

## Where things are stored

| What | Where |
|---|---|
| Runs, params, metrics, tags, the registry | Database `mlflow` on the compose Postgres |
| Model files and other artifacts | `STOA_MLFLOW_ARTIFACTS`: the `mlflow-artifacts` Docker volume by default, or a `gs://` path (the server uses the `.secrets/` service-account key) |

The server uploads and downloads artifacts itself (`--serve-artifacts`), so a client needs only the server's address, never bucket credentials.

**For now each machine has its own server**, so runs stay on the machine that made them. The plan is one shared server on a small GCP VM with the rest of the stack (CLAUDE.md, "Infrastructure"); teammates then reach it through an SSH tunnel and set `MLFLOW_TRACKING_URI`, and nothing in the training code changes.
