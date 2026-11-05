import json

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models import tracking


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    """A throwaway MLflow store in a folder, so tests need no server."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    yield tmp_path
    mlflow.end_run()


def write_pointer(path, md5):
    path.write_text(f"outs:\n- md5: {md5}\n  path: {path.stem}\n")


def test_dvc_versions_reads_each_pointer(tmp_path):
    write_pointer(tmp_path / "raw.dvc", "aaa.dir")
    write_pointer(tmp_path / "processed.dvc", "ccc.dir")
    assert tracking.dvc_versions(tmp_path) == {
        "stoa.dvc.raw": "aaa.dir",
        "stoa.dvc.interim": "missing",
        "stoa.dvc.processed": "ccc.dir",
    }


def test_frame_spec_tags(tmp_path):
    spec = tmp_path / "frame_spec.json"
    assert tracking.frame_spec_tags(spec) == {"stoa.frame_spec": "missing"}
    spec.write_text(
        json.dumps({"config_hash": "h1", "dvc_data_version": "v1", "rows": 10})
    )
    tags = tracking.frame_spec_tags(spec)
    assert tags["stoa.frame_spec"] == "logged"
    assert tags["stoa.frame.config_hash"] == "h1"
    assert tags["stoa.frame.dvc_data_version"] == "v1"
    assert tags["stoa.frame.rows"] == "10"


def test_tracking_uri_prefers_the_environment(monkeypatch):
    cfg = {"tracking": {"uri": "http://localhost:5001"}}
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.tracking_uri(cfg) == "http://localhost:5001"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert tracking.tracking_uri(cfg) == "http://mlflow:5000"


def test_check_server_fails_fast_without_a_server():
    with pytest.raises(tracking.ServerUnreachable, match="docker compose up -d mlflow"):
        tracking.check_server("http://127.0.0.1:9", timeout_s=0.5)


def test_run_records_provenance_params_and_frame_spec(local_store, monkeypatch):
    spec = local_store / "frame_spec.json"
    spec.write_text(json.dumps({"config_hash": "h1", "dvc_data_version": "v1"}))
    monkeypatch.setattr(tracking, "frame_spec_path", lambda site: spec)

    with tracking.run(
        "t", params={"max_iter": 5}, experiment="test", tags={"who": "me"}
    ) as r:
        mlflow.log_metric("wape_cv", 0.4)

    got = mlflow.get_run(r.info.run_id)
    assert got.data.params == {"max_iter": "5"}
    assert got.data.metrics == {"wape_cv": 0.4}
    assert got.data.tags["who"] == "me"
    assert got.data.tags["stoa.frame.config_hash"] == "h1"
    for key in ["stoa.git_commit", "stoa.git_dirty", "stoa.dvc.processed", "stoa.site"]:
        assert key in got.data.tags
    names = [a.path for a in mlflow.artifacts.list_artifacts(run_id=r.info.run_id)]
    assert names == ["data"]


def test_run_without_frame_spec_still_logs(local_store, monkeypatch):
    monkeypatch.setattr(
        tracking, "frame_spec_path", lambda site: local_store / "none.json"
    )
    with tracking.run("t", experiment="test") as r:
        pass
    assert mlflow.get_run(r.info.run_id).data.tags["stoa.frame_spec"] == "missing"


def test_logged_hgb_model_reloads_without_extra_arguments(local_store):
    rng = np.random.default_rng(0)
    x = rng.random((200, 3))
    y = rng.poisson(5, 200)
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=5).fit(x, y)
    with tracking.run("t", experiment="test"):
        info = tracking.log_sklearn_model(model, input_example=x[:2])
    reloaded = mlflow.sklearn.load_model(info.model_uri)
    assert np.allclose(reloaded.predict(x[:5]), model.predict(x[:5]))
