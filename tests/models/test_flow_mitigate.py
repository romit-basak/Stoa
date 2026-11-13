"""Bias mitigation on synthetic data: every mitigation is retrained or calibrated and scored."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.flow import mitigate
from tests.models.conftest import make_data
from tests.models.test_flow_pipeline import run_all

BIAS_CFG = {
    "errors": {"bootstrap_reps": 20, "seed": 0, "worse_margin": 0.0},
    "thin_min_sensors": 2,
}


def fake_slices(keys: pd.DataFrame):
    """Volume tiers by sensor, standing in for the slices src/bias builds from disk."""
    # Every fourth sensor is quiet, so no fold trains on as many quiet as busy sensors.
    tier = np.where(keys["location_id"] % 4 == 0, "quiet", "busy")
    rows = keys.assign(volume_tier=tier).reset_index(drop=True)
    return rows, ["volume_tier"], BIAS_CFG


def test_every_mitigation_is_scored_on_the_same_hours(tmp_path, fcfg, mlflow_store):
    out = tmp_path / "models"
    run_all(fcfg, out, families=["hgb"])
    table = mitigate.mitigate_variant(
        make_data("rich"), fcfg, out / "rich", fake_slices
    )

    names = [m["name"] for m in fcfg["mitigation"]]
    assert table["mitigation"].tolist() == names
    assert np.isfinite(table[["wape_cv", "worst_slice_ratio"]].to_numpy()).all()
    # "none" is the selected model as trained: the same fold score as its candidate run.
    sel = pd.read_json(out / "rich" / "candidates.json", typ="series")
    none = table.set_index("mitigation").loc["none"]
    assert none["wape_cv"] == pytest.approx(sel["models"]["hgb"]["metrics"]["wape_cv"])
    # Every synthetic sensor has as many rows, so sensor_equal weights are all 1; the tiers
    # aren't equal, so that weighting changes the model.
    assert (
        table.set_index("mitigation").loc["volume_tier_equal", "wape_cv"]
        != none["wape_cv"]
    )
    assert table.set_index("mitigation").loc["calibrated", "wape_cv"] < none["wape_cv"]
    art = out / "rich" / "mitigation"
    assert {"summary.csv", "slices.csv", "mitigation.png"} <= {
        p.name for p in art.iterdir()
    }
    slices = pd.read_csv(art / "slices.csv")
    assert set(slices["mitigation"]) == set(names)


def test_fit_passes_row_weights(data, fcfg):
    from src.models.flow import candidates

    rows = np.flatnonzero(data.fold >= 0)
    x, y = data.frame.matrix(rows), data.y[rows]
    params = candidates.capped("lgbm", fcfg["candidates"]["lgbm"]["params"], 5)
    plain = candidates.fit(candidates.make("lgbm", params, 7), x, y)
    w = np.where(y > np.median(y), 10.0, 0.1)
    weighted = candidates.fit(candidates.make("lgbm", params, 7), x, y, sample_weight=w)
    assert not np.allclose(
        candidates.predict(plain, x), candidates.predict(weighted, x)
    )
