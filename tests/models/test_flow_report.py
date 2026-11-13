"""The training report: loss functions, cost curves and the parameter counts per family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.flow import candidates, report


def test_poisson_deviance_is_zero_for_a_perfect_prediction_and_positive_otherwise():
    y = np.array([0.0, 3.0, 10.0])
    assert report.poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-5)
    assert report.poisson_deviance(y, y + 1) > 0
    # One row, y = 2, mu = 1: 2 * (2 log 2 - 1)
    assert report.poisson_deviance([2.0], [1.0]) == pytest.approx(
        2 * (2 * np.log(2) - 1)
    )


def test_pinball_weights_misses_by_side():
    assert report.pinball([10.0], [8.0], 0.9) == pytest.approx(0.9 * 2)
    assert report.pinball([10.0], [12.0], 0.9) == pytest.approx(0.1 * 2)


@pytest.mark.parametrize("family", candidates.FAMILIES)
def test_cost_falls_from_the_first_tree_to_the_last(family, data, fcfg):
    rows = np.flatnonzero(data.fold >= 0)
    x, y = data.frame.matrix(rows), data.y[rows]
    params = candidates.capped(family, fcfg["candidates"][family]["params"], 20)
    model = candidates.fit(candidates.make(family, params, 7), x, y)
    curve = report.cost_curve(family, model, {"training rows": (x, y)}, float(y.mean()))
    assert curve["trees"].iloc[0] == 0 and curve["trees"].iloc[-1] == report.n_trees(
        family, model
    )
    start, end = curve["training rows"].iloc[[0, -1]]
    assert start == pytest.approx(report.poisson_deviance(y, np.full(len(y), y.mean())))
    assert end < start
    counts = report.learned(family, model)
    assert counts["trees"] == report.n_trees(family, model)
    assert counts["leaves"] > counts["trees"] and counts["features_used"] >= 1


def test_render_has_every_section_and_start_and_end():
    facts = report.DataFacts(
        ("2009-05-01", "2026-11-12"), 1000, 10, 8, {"holdout": [1, 3]}
    )
    curve = pd.DataFrame({"trees": [0, 5, 10], "training rows": [4.0, 2.0, 1.0]})
    text = report.render(
        "m",
        "A model.",
        facts,
        "Poisson deviance",
        "mean",
        {"Cost": curve},
        {"trees": 10},
        {"learning_rate": 0.1},
        report.tuning_text(False, None),
    )
    assert "| training rows | 4 | 1 | -75.0% |" in text
    assert "Never seen, holdout: 2 sensors (1, 3)" in text
    assert "`learning_rate`: 0.1" in text and "not tuned" in text
