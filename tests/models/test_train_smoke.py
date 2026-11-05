"""Smoke test: run_variant with tiny synthetic data, mocked MLflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.flow.train import run_variant


def _make_config(tmp_path: Path) -> dict:
    return {
        "training": {"folds": 2, "seed": 0, "calibration_days": 7},
        "default_params": {
            "max_iter": 10,
            "learning_rate": 0.1,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
            "random_state": 0,
        },
        "gates": {
            "mae_log_improvement_pct": 15.0,
            "interval_coverage_low": 0.75,
            "interval_coverage_high": 0.85,
        },
    }


def test_run_variant_returns_metrics(tiny_hourly, tiny_osm, tmp_path):
    cfg = _make_config(tmp_path)
    # Row columns used by tiny_hourly (a minimal subset)
    result = run_variant(
        hourly=tiny_hourly,
        osm=tiny_osm,
        cfg=cfg,
        variant="rich",
        n_iter=10,
        folds=2,
        out_dir=tmp_path / "flow" / "rich",
    )
    assert "wape" in result
    assert "mae_log" in result
    assert "coverage" in result
    assert isinstance(result["wape"], float)
    assert 0.0 <= result["coverage"] <= 1.0


def test_run_variant_writes_oof_parquet(tiny_hourly, tiny_osm, tmp_path):
    cfg = _make_config(tmp_path)
    out = tmp_path / "flow" / "rich"
    run_variant(
        hourly=tiny_hourly,
        osm=tiny_osm,
        cfg=cfg,
        variant="rich",
        n_iter=10,
        folds=2,
        out_dir=out,
    )
    oof_path = out / "oof_predictions.parquet"
    assert oof_path.exists(), "oof_predictions.parquet was not written"
    df = pd.read_parquet(oof_path)
    assert "oof_mean" in df.columns
    assert len(df) == len(tiny_hourly)


def test_run_variant_writes_model_pkl(tiny_hourly, tiny_osm, tmp_path):
    cfg = _make_config(tmp_path)
    out = tmp_path / "flow" / "rich"
    run_variant(
        hourly=tiny_hourly,
        osm=tiny_osm,
        cfg=cfg,
        variant="rich",
        n_iter=10,
        folds=2,
        out_dir=out,
    )
    assert (out / "model.pkl").exists()
