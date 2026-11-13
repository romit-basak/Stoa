"""Log model training runs to MLflow, with the data and code version they were trained on.

Every training script wraps its work in `tracking.run(...)`:

    from src.models import tracking

    with tracking.run("hgb_poisson", params={"max_iter": 300, "learning_rate": 0.1}):
        model.fit(X, y)
        mlflow.log_metrics({"wape_cv": 0.41, "mae_cv": 118.0})
        tracking.log_sklearn_model(model, input_example=X[:5])

The run then records, without the caller doing anything:
- tags `stoa.git_commit` / `stoa.git_dirty`: the code it ran (dirty = uncommitted changes);
- tags `stoa.dvc.raw|interim|processed`: the DVC data version of each folder (the md5 in
  `data/*.dvc`), so `dvc checkout` at that commit brings back the exact data;
- the training frame's `frame_spec.json` as an artifact, and its config hash and OSM extract as
  tags (CLAUDE.md, "MLOps lifecycle": every run records its DVC data version).

Server: the `mlflow` service in docker-compose.yaml (http://localhost:5001). Set
MLFLOW_TRACKING_URI to log somewhere else, and MLFLOW_EXPERIMENT_NAME to log to another
experiment than `tracking.experiment` (the gate then reads that experiment too).

Usage: python -m src.models.tracking --smoke    # log a tiny model to check the setup
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import requests
import yaml

from src.features.common import load_features_config
from src.ingest.common import REPO_ROOT

CONFIG = REPO_ROOT / "configs" / "models.yaml"
DVC_FOLDERS = ("raw", "interim", "processed")


class ServerUnreachable(RuntimeError):
    pass


def load_models_config(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def tracking_uri(cfg: Mapping[str, Any] | None = None) -> str:
    """MLFLOW_TRACKING_URI if set, else the configured server."""
    cfg = cfg or load_models_config()
    return os.environ.get("MLFLOW_TRACKING_URI") or cfg["tracking"]["uri"]


def experiment_name(cfg: Mapping[str, Any] | None = None) -> str:
    """MLFLOW_EXPERIMENT_NAME if set (to keep side studies out of the main table), else the
    configured experiment."""
    cfg = cfg or load_models_config()
    return os.environ.get("MLFLOW_EXPERIMENT_NAME") or cfg["tracking"]["experiment"]


def check_server(uri: str, timeout_s: float = 3.0) -> None:
    """Fail at once with a clear message instead of MLflow's minutes of retries."""
    if not uri.startswith("http"):
        return
    try:
        requests.get(uri.rstrip("/") + "/health", timeout=timeout_s).raise_for_status()
    except requests.RequestException as e:
        raise ServerUnreachable(
            f"No MLflow server at {uri}. Start it with `docker compose up -d mlflow`, "
            f"or set MLFLOW_TRACKING_URI. ({e.__class__.__name__})"
        ) from None


def git_state(repo: Path = REPO_ROOT) -> dict[str, str]:
    """The commit and whether the working tree has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"stoa.git_commit": "unknown", "stoa.git_dirty": "unknown"}
    return {"stoa.git_commit": commit, "stoa.git_dirty": str(dirty).lower()}


def dvc_versions(data_dir: Path = REPO_ROOT / "data") -> dict[str, str]:
    """The md5 of each DVC-tracked data folder, from its pointer file."""
    tags = {}
    for name in DVC_FOLDERS:
        pointer = data_dir / f"{name}.dvc"
        md5 = "missing"
        if pointer.exists():
            md5 = yaml.safe_load(pointer.read_text())["outs"][0]["md5"]
        tags[f"stoa.dvc.{name}"] = md5
    return tags


def training_site() -> str:
    """The site models train on: the one the hourly table is built for."""
    return load_features_config()["counts_hourly"]["site"]


def frame_spec_path(site: str) -> Path:
    fcfg = load_features_config()
    return REPO_ROOT / fcfg["processed_dir"] / "training" / site / "frame_spec.json"


def frame_spec_tags(spec_path: Path) -> dict[str, str]:
    """The training frame's settings and inputs, or a marker that it hasn't been built."""
    if not spec_path.exists():
        return {"stoa.frame_spec": "missing"}
    spec = json.loads(spec_path.read_text())
    return {
        "stoa.frame_spec": "logged",
        "stoa.frame.config_hash": str(spec.get("config_hash")),
        "stoa.frame.dvc_data_version": str(spec.get("dvc_data_version")),
        "stoa.frame.osm_pbf_sha256": str(spec.get("osm_pbf_sha256")),
        "stoa.frame.rows": str(spec.get("rows")),
        "stoa.frame.sensors": str(spec.get("sensors")),
    }


@contextmanager
def run(
    run_name: str,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    experiment: str | None = None,
    site: str | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run with Stoa's provenance tags; log metrics and models inside it."""
    cfg = load_models_config()
    uri = tracking_uri(cfg)
    check_server(uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or experiment_name(cfg))
    site = site or training_site()
    spec = frame_spec_path(site)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags(
            {
                "stoa.site": site,
                **git_state(),
                **dvc_versions(),
                **frame_spec_tags(spec),
                **(tags or {}),
            }
        )
        if params:
            mlflow.log_params(dict(params))
        if spec.exists():
            mlflow.log_artifact(str(spec), artifact_path="data")
        yield active


def log_sklearn_model(
    model: Any,
    name: str = "model",
    input_example: Any = None,
    registered_model_name: str | None = None,
) -> Any:
    """Log a scikit-learn model in the safe skops format, loadable without extra arguments.

    skops refuses to load types it doesn't know unless they're listed as trusted. We trained
    the model ourselves, so its own scikit-learn types are listed when it's saved (and only
    those: a type from any other package still has to be trusted by hand).
    """
    import mlflow.sklearn
    import skops.io

    trusted = [
        t
        for t in skops.io.get_untrusted_types(data=skops.io.dumps(model))
        if t.startswith("sklearn.")
    ]
    return mlflow.sklearn.log_model(
        model,
        name=name,
        input_example=input_example,
        registered_model_name=registered_model_name,
        skops_trusted_types=trusted or None,
    )


def smoke() -> str:
    """Train a tiny model on random numbers and log it, to check the server end to end."""
    import mlflow.sklearn
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    x = rng.random((500, 4))
    y = rng.poisson(20 * x[:, 0] + 5)
    params = {"loss": "poisson", "max_iter": 20}
    tags = {"stoa.smoke": "true"}
    with run("smoke", params=params, experiment="smoke-test", tags=tags) as active:
        model = HistGradientBoostingRegressor(**params).fit(x, y)
        pred = model.predict(x)
        mlflow.log_metric("wape_train", float(np.abs(pred - y).sum() / y.sum()))
        info = log_sklearn_model(model, input_example=x[:5])
    reloaded = mlflow.sklearn.load_model(info.model_uri)
    assert np.allclose(reloaded.predict(x[:5]), pred[:5])
    return active.info.run_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--smoke", action="store_true", help="log a tiny model to check the setup"
    )
    args = ap.parse_args()
    if not args.smoke:
        ap.print_help()
        return
    run_id = smoke()
    print(
        f"Logged and reloaded smoke run {run_id} at {tracking_uri()} (experiment smoke-test)"
    )


if __name__ == "__main__":
    main()
