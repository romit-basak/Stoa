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

    # The bundle: mean and widened interval, from a DataFrame of the feature columns.
    bundle = mlflow.pyfunc.load_model(sel["bundle_uri"])
    cols = make_data().frame.columns
    frame = pd.DataFrame(make_data().frame.matrix(np.arange(5)), columns=cols)
    out = bundle.predict(frame[cols[::-1]])  # column order doesn't matter
    assert list(out.columns) == ["predicted", "lower", "upper"] and len(out) == 5
    assert (out["lower"] <= out["predicted"]).all()
    assert (out["predicted"] <= out["upper"]).all()


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
    bundle_uri = client.get_run(g.run_id).data.tags["stoa.bundle_uri"]
    assert client.get_model_version(name, first["version"]).source == bundle_uri


def test_registry_keeps_a_better_champion_and_rolls_back_step_by_step(
    tmp_path, fcfg, mlflow_store
):
    from dataclasses import replace

    import pytest

    run_all(fcfg, tmp_path / "models", families=["hgb"])
    mlflow.set_tracking_uri(mlflow_store)
    client = MlflowClient()
    exp = client.get_experiment_by_name(
        tracking.load_models_config()["tracking"]["experiment"]
    )
    g = registry.gate(client, exp.experiment_id, "rich", fcfg["gates"])
    name = registry.model_name("rich")
    v1 = registry.register(client, replace(g, improvement=0.15), promote=True)
    v2 = registry.register(client, replace(g, improvement=0.16), promote=True)
    v3 = registry.register(
        client, replace(g, improvement=0.155), promote=True, tolerance=0.01
    )
    assert v3["alias"] == "champion"  # 0.5 points worse, within the tolerance

    # Two points worse than the champion: registered, but the champion stays.
    worse = registry.register(
        client, replace(g, improvement=0.135), promote=True, tolerance=0.01
    )
    assert worse["held_back"] and worse["alias"] == "challenger"
    assert registry.alias_version(client, name, "champion") == v3["version"]
    assert registry.alias_version(client, name, "challenger") == worse["version"]

    # Rollback walks back one version at a time: v3 -> v2 -> v1, then refuses.
    registry.rollback(client, "rich")
    assert registry.alias_version(client, name, "champion") == v2["version"]
    assert registry.alias_version(client, name, "previous") == v1["version"]
    registry.rollback(client, "rich")
    assert registry.alias_version(client, name, "champion") == v1["version"]
    assert registry.alias_version(client, name, "previous") is None

    # Rolling back both variants changes neither when one has nothing to roll back to.
    registry.register(client, replace(g, improvement=0.2), promote=True)
    with pytest.raises(SystemExit, match="nothing rolled back"):
        registry.rollback_all(client, ["rich", "portable"])
    assert registry.alias_version(client, name, "previous") == v1["version"]
