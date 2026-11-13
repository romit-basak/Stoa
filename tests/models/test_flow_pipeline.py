"""The whole flow-model path on synthetic data: train, select, gate, register, roll back.

MLflow logs to a file store in a temporary folder (no server needed: tracking.check_server
skips non-HTTP URIs). The bias check is replaced by a stand-in, because the real one builds
the slices from the full hourly table on disk.
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.models import tracking
from src.models.flow import registry, select, train
from src.models.flow.candidates import FAMILIES
from tests.models.conftest import make_data


def fake_bias(oof: pd.DataFrame, names: list[str], skip: list[str]):
    worst = pd.DataFrame(
        {
            "model": names,
            "worst_slice_ratio": np.linspace(1.5, 2.0, len(names)),
            "worst_slice": ["time_of_day=night"] * len(names),
            "groups_worse": [1] * len(names),
        }
    )
    return pd.DataFrame({"model": names, "dimension": "x", "group": "y"}), worst


def run_all(fcfg, out, families=FAMILIES):
    summaries = train.run(
        ["rich", "portable"],
        list(families),
        fcfg,
        out,
        load_data=make_data,
        max_trees=5,
    )
    selected = {
        v: select.select_variant(make_data(v), fcfg, out / v, bias_fn=fake_bias)
        for v in ("rich", "portable")
    }
    return summaries, selected


def test_train_and_select(tmp_path, fcfg, mlflow_store):
    out = tmp_path / "models"
    summaries, _ = run_all(fcfg, out)

    rich = summaries["rich"]
    assert set(rich["models"]) == set(FAMILIES)
    assert set(rich["baselines"]) == {"baseline_betweenness", "baseline_space_syntax"}
    # The baselines ran once and both variants point at the same runs.
    assert rich["baselines"] == summaries["portable"]["baselines"]
    oof = pd.read_parquet(out / "rich" / "oof.parquet")
    assert {"y", "fold", *FAMILIES, "baseline_space_syntax"} <= set(oof.columns)
    assert (
        oof["fold"] >= 0
    ).all()  # holdout precincts are never in the out-of-fold rows

    sel = json.loads((out / "rich" / "selected.json").read_text())
    assert sel["family"] in FAMILIES
    assert {
        "log_mae_cv",
        "log_mae_cv_cal",
        "coverage80_cv",
        "coverage80_fold0",
        "log_mae_holdout",
        "wape_holdout",
        "worst_slice_ratio",
        "log_mae_other_days",
    } <= set(sel["metrics"])
    for name in ("ranking.csv", "model_comparison.png", "calibration_curve.png"):
        assert (out / "rich" / "selected" / name).exists()

    # A training report for every model trained, and one for the selected model.
    reports = out / "rich" / "reports"
    assert {f"{f}_rich.md" for f in FAMILIES} <= {p.name for p in reports.iterdir()}
    assert (reports / f"selected_{sel['family']}_rich.md").exists()
    assert {p.name for p in (out / "baselines" / "reports").iterdir()} == {
        "baseline_betweenness.md",
        "baseline_space_syntax.md",
    }
    text = (reports / "lgbm_rich.md").read_text()
    for heading in (
        "## Data",
        "## Loss and cost",
        "## Learned parameters",
        "## Hyperparameters",
    ):
        assert heading in text
    assert "Never seen, holdout precincts" in text and "Poisson deviance" in text

    client = MlflowClient(mlflow_store)
    exp = client.get_experiment_by_name(
        tracking.load_models_config()["tracking"]["experiment"]
    )
    runs = client.search_runs([exp.experiment_id], max_results=100)
    names = {r.info.run_name for r in runs}
    assert {"hgb_rich", "lgbm_portable", "xgb_rich", "xgb_portable"} <= names
    assert "baseline_space_syntax" in names
    roles = [r.data.tags["stoa.role"] for r in runs]
    assert roles.count("selected") == 2 and roles.count("baseline") == 2
    for r in runs:
        files = {a.path for a in client.list_artifacts(r.info.run_id)}
        assert "model_report.md" in files, r.info.run_name

    model = mlflow.pyfunc.load_model(sel["model_uris"]["model"])
    x = np.zeros((2, len(make_data().frame.columns)), dtype="float32")
    assert model.predict(x).shape == (2,)


def test_registry_aliases_and_rollback(tmp_path, fcfg, mlflow_store):
    run_all(fcfg, tmp_path / "models", families=["hgb"])
    mlflow.set_tracking_uri(mlflow_store)
    client = MlflowClient()
    exp = client.get_experiment_by_name(
        tracking.load_models_config()["tracking"]["experiment"]
    )
    g = registry.gate(client, exp.experiment_id, "rich", fcfg["gates"])
    assert g.run_id and g.baseline_run_id  # found the matching Space Syntax run

    first = registry.register(client, g, promote=True)
    second = registry.register(client, g, promote=True)
    name = registry.model_name("rich")
    assert registry.alias_version(client, name, "champion") == second["version"]
    assert registry.alias_version(client, name, "previous") == first["version"]

    back = registry.rollback(client, "rich")
    assert back["champion"] == first["version"]
    assert registry.alias_version(client, name, "champion") == first["version"]
    assert registry.alias_version(client, name, "previous") is None
