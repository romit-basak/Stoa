"""The candidate architectures for the flow model: a Poisson objective for the mean, and a
quantile objective of the same family for the prediction interval.

- hgb: scikit-learn HistGradientBoostingRegressor
- lgbm: LightGBM
- xgb: XGBoost (histogram trees)

All three handle missing values (NaN) in the features themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FAMILIES = ("hgb", "lgbm", "xgb")
# The size parameter each family's --max-trees cap applies to.
TREES_PARAM = {"hgb": "max_iter", "lgbm": "n_estimators", "xgb": "n_estimators"}


def make(
    family: str, params: dict[str, Any], seed: int, quantile: float | None = None
) -> Any:
    """An unfitted estimator: the Poisson mean model, or the `quantile` model."""
    if family == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        loss = (
            {"loss": "quantile", "quantile": quantile}
            if quantile
            else {"loss": "poisson"}
        )
        return HistGradientBoostingRegressor(random_state=seed, **loss, **params)
    if family == "lgbm":
        from lightgbm import LGBMRegressor

        obj = {"objective": "quantile", "alpha": quantile} if quantile else {}
        return LGBMRegressor(
            **{"objective": "poisson", **obj},
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    if family == "xgb":
        from xgboost import XGBRegressor

        obj = (
            {"objective": "reg:quantileerror", "quantile_alpha": quantile}
            if quantile
            else {"objective": "count:poisson"}
        )
        return XGBRegressor(
            **obj, tree_method="hist", random_state=seed, n_jobs=-1, **params
        )
    raise ValueError(f"unknown family {family!r}: one of {FAMILIES}")


def capped(
    family: str, params: dict[str, Any], max_trees: int | None
) -> dict[str, Any]:
    """params with the number of trees or boosting rounds capped (quick runs)."""
    if not max_trees:
        return dict(params)
    key = TREES_PARAM[family]
    return {**params, key: min(int(params.get(key, max_trees)), max_trees)}


def predict(model: Any, x: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """Non-negative predictions, in chunks to bound memory on millions of rows."""
    parts = [model.predict(x[i : i + chunk]) for i in range(0, len(x), chunk)]
    out = np.concatenate(parts) if parts else np.empty(0)
    return np.maximum(out, 0.0).astype("float32")
