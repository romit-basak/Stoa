"""The selected flow model with its 80% interval, as one MLflow pyfunc model.

`select` fits three models: the mean (Poisson) and the 10% and 90% quantiles. The quantile
interval is too narrow at a new location, so prediction widens it by margins measured on
out-of-fold misses (`select.interval_margins`). This bundle holds the three models and the
margins, so the registered version predicts what CLAUDE.md's Predictions interface needs:
`predicted`, `lower` and `upper` per row (segment-hour). The bounds are also stretched to
include `predicted` where the mean model falls outside them, which can only raise coverage.

Load it like any MLflow model:

    model = mlflow.pyfunc.load_model("models:/stoa-flow-rich@champion")
    out = model.predict(features)   # a DataFrame with the training frame's feature columns
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd


def widen(
    low: np.ndarray, high: np.ndarray, margins: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Widen a quantile interval by (lower, upper) margins on log1p counts; a negative
    margin narrows that side."""
    lower, upper = margins
    low = np.maximum(np.expm1(np.log1p(low) - lower), 0.0)
    return low, np.expm1(np.log1p(high) + upper)


class FlowBundle(mlflow.pyfunc.PythonModel):
    """Mean model, 10% and 90% quantile models, and the interval margins."""

    def __init__(
        self,
        mean: Any,
        low: Any,
        high: Any,
        margins: tuple[float, float],
        columns: list[str],
    ) -> None:
        self.mean, self.low, self.high = mean, low, high
        self.margins = (float(margins[0]), float(margins[1]))
        self.columns = list(columns)

    def _matrix(self, model_input: Any) -> np.ndarray:
        if isinstance(model_input, pd.DataFrame):
            missing = [c for c in self.columns if c not in model_input.columns]
            if missing:
                raise ValueError(f"missing feature columns: {missing[:5]}")
            model_input = model_input[self.columns]
        return np.asarray(model_input, dtype="float32")

    # No type hints here: MLflow reads them as an input schema, and these aren't one.
    def predict(self, context, model_input, params=None):
        from src.models.flow import candidates

        x = self._matrix(model_input)
        mean = candidates.predict(self.mean, x)
        low, high = widen(
            candidates.predict(self.low, x),
            candidates.predict(self.high, x),
            self.margins,
        )
        return pd.DataFrame(
            {
                "predicted": mean,
                "lower": np.minimum(low, mean),
                "upper": np.maximum(high, mean),
            }
        )
