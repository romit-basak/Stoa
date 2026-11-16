"""Selection follows the train run's settings: here, its seed (train --seed)."""

from __future__ import annotations

from src.models.flow import select, train
from tests.models.conftest import make_data
from tests.models.test_flow_pipeline import fake_bias


def test_selection_uses_the_train_runs_seed(tmp_path, fcfg, mlflow_store, monkeypatch):
    out = tmp_path / "models"
    train.run(
        ["rich"],
        ["hgb"],
        {**fcfg, "seed": fcfg["seed"] + 4},
        out,
        load_data=make_data,
        max_trees=5,
    )
    seeds = []
    fit_final = select.fit_final

    def recording(*args):
        seeds.append(args[-1])
        return fit_final(*args)

    monkeypatch.setattr(select, "fit_final", recording)
    select.select_variant(make_data("rich"), fcfg, out / "rich", bias_fn=fake_bias)
    assert seeds == [fcfg["seed"] + 4]
