"""src.models.flow.figures against a throwaway MLflow store."""

import json
import time

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.models.flow import figures

PNG = (  # a 1x1 PNG, enough to stand in for a chart
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00"
    b"\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
    b"\x00\x00\x00IEND\xaeB`\x82"
)


def _tick():
    time.sleep(0.01)  # distinct start times, since runs are ordered by them


def _tuning(
    fam: str, variant: str = "rich", commit: str = "abc123", dvc: str = "v1"
) -> None:
    with mlflow.start_run(run_name=f"tune_{fam}_{variant}"):
        mlflow.set_tags(
            {
                "stoa.role": "tuning",
                "stoa.family": fam,
                "stoa.variant": variant,
                "stoa.git_commit": commit,
                "stoa.dvc.processed": dvc,
            }
        )
        for i, loss in enumerate([0.95, 0.90, 0.92]):
            with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                mlflow.log_metric("log_mae_inner", loss)
        mlflow.log_params({"best_learning_rate": 0.05, "best_max_iter": 400})
        mlflow.log_metric("log_mae_inner_best", 0.90)
        mlflow.log_dict(
            {"learning_rate": 0.7, "max_iter": 0.3}, "param_importance.json"
        )
    _tick()


def _selected(tmp_path, variant: str = "rich", dvc: str = "v1") -> str:
    art = tmp_path / f"selection_{variant}"
    art.mkdir()
    for name in figures.SELECTION_CHARTS:
        (art / f"{name}.png").write_bytes(PNG)
    with mlflow.start_run(run_name=f"lgbm_{variant}") as run:
        mlflow.set_tags(
            {
                "stoa.role": "selected",
                "stoa.variant": variant,
                "stoa.dvc.processed": dvc,
                "stoa.git_commit": "abc123",
            }
        )
        mlflow.log_param("family", "lgbm")
        mlflow.log_metrics(
            {"log_mae_cv": 0.86, "coverage80_cv": 0.78, "worst_slice_ratio": 1.9}
        )
        mlflow.log_artifacts(str(art), artifact_path="selection")
    _tick()
    return run.info.run_id


def _mitigation(name: str, wape: float, dvc: str = "v1", variant: str = "rich") -> None:
    with mlflow.start_run(run_name=f"mitigation_{name}_{variant}"):
        mlflow.set_tags(
            {
                "stoa.role": "mitigation",
                "stoa.mitigation": name,
                "stoa.variant": variant,
                "stoa.dvc.processed": dvc,
            }
        )
        mlflow.log_metrics(
            {"wape_cv": wape, "log_mae_cv": 0.9, "worst_slice_ratio": 1.8}
        )
    _tick()


@pytest.fixture
def exp(mlflow_store):
    mlflow.set_tracking_uri(mlflow_store)
    mlflow.set_experiment("figures-test")
    client = MlflowClient()
    return client, client.get_experiment_by_name("figures-test").experiment_id


def test_tuned_training_exports_every_chart(tmp_path, exp):
    client, exp_id = exp
    for fam in ("hgb", "lgbm"):
        _tuning(fam)
    sel = _selected(tmp_path)
    _mitigation("none", 0.64)
    _mitigation("calibrated", 0.40)
    _mitigation("stale", 0.99, dvc="v0")  # another data version: left out

    out = tmp_path / "figures"
    out.mkdir()
    s = figures.export_variant(client, exp_id, "rich", out)

    assert s["selected_run_id"] == sel and s["family"] == "lgbm"
    assert s["git_commit"] == "abc123" and s["data_version"] == "v1"
    assert s["missing"] == []
    for name in s["written"]:
        assert (out / name).stat().st_size > 0, name
    assert {m["mitigation"] for m in s["mitigations"]} == {"none", "calibrated"}
    assert s["tuning"]["hgb"]["trials"] == 3
    assert s["tuning"]["lgbm"]["params"] == {"learning_rate": "0.05", "max_iter": "400"}
    assert s["metrics"]["log_mae_cv"] == pytest.approx(0.86)


def test_untuned_training_skips_tuning_charts(tmp_path, exp):
    client, exp_id = exp
    _selected(tmp_path)
    out = tmp_path / "figures"
    out.mkdir()
    s = figures.export_variant(client, exp_id, "rich", out)
    assert set(s["missing"]) == {
        "mitigation_rich.png",
        "hyperparameter_importance_rich.png",
        "tuning_history_rich.png",
    }
    assert not (out / "tuning_history_rich.png").exists()
    assert s["tuning"] == {}


def test_tuning_after_the_selected_run_belongs_to_the_next_training(tmp_path, exp):
    client, exp_id = exp
    _selected(tmp_path)
    _tuning("hgb")  # started after selection: a later training's search
    out = tmp_path / "figures"
    out.mkdir()
    assert figures.export_variant(client, exp_id, "rich", out)["tuning"] == {}


def test_no_selected_run(tmp_path, exp):
    client, exp_id = exp
    s = figures.export_variant(client, exp_id, "portable", tmp_path)
    assert s == {"variant": "portable", "error": "no selected run"}
    json.dumps(s)


def test_a_chart_missing_from_the_run_is_listed(tmp_path, exp):
    client, exp_id = exp
    run_id = _selected(tmp_path)
    client.delete_tag(run_id, "stoa.git_commit")  # older runs may lack tags too
    assert not figures.download(
        run_id, "selection/no_such_chart.png", tmp_path / "x.png"
    )
    assert figures.load_json_artifact(run_id, "param_importance.json") is None
    s = figures.export_variant(client, exp_id, "rich", tmp_path / "out")
    assert s["git_commit"] is None
    assert "model_comparison_rich.png" in s["written"]


def test_tuning_from_other_code_is_not_attached(tmp_path, exp):
    """A tuned training on older code, then an untuned one on newer code: the newer
    selected run gets no tuning charts."""
    client, exp_id = exp
    _tuning("lgbm", commit="old000")
    _selected(tmp_path)  # git commit abc123
    out = tmp_path / "figures"
    s = figures.export_variant(client, exp_id, "rich", out)
    assert s["tuning"] == {}
    assert "tuning_history_rich.png" in s["missing"]
