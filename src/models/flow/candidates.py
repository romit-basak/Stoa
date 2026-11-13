"""The candidate architectures for the flow model: a Poisson objective for the mean, and a
quantile objective for the prediction interval.

- hgb: scikit-learn HistGradientBoostingRegressor
- lgbm: LightGBM
- xgb: XGBoost (histogram trees)
- rf: scikit-learn RandomForestRegressor (Poisson splitting)

A boosting family's interval models use its own quantile objective. scikit-learn's random
forest has no quantile loss, so rf's interval models are HistGradientBoosting quantile
models with rf's depth and leaf size; select.py widens them as it does every family's.

All four handle missing values (NaN) in the features themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FAMILIES = ("hgb", "lgbm", "xgb", "rf")
# The size parameter each family's --max-trees cap applies to.
TREES_PARAM = {
    "hgb": "max_iter",
    "lgbm": "n_estimators",
    "xgb": "n_estimators",
    "rf": "n_estimators",
}
# rf settings that its interval models (HistGradientBoosting) also take.
RF_INTERVAL_PARAMS = ("max_depth", "min_samples_leaf")


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
    if family == "rf":
        if quantile:
            from sklearn.ensemble import HistGradientBoostingRegressor

            shared = {k: params[k] for k in RF_INTERVAL_PARAMS if k in params}
            return HistGradientBoostingRegressor(
                loss="quantile", quantile=quantile, random_state=seed, **shared
            )
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            criterion="poisson", random_state=seed, n_jobs=-1, **params
        )
    raise ValueError(f"unknown family {family!r}: one of {FAMILIES}")


def fit(model: Any, x: np.ndarray, y: np.ndarray) -> Any:
    """Fit `model`, with any feature that has no values in these rows set to a constant.

    scikit-learn 1.9 crashes HistGradientBoosting on an all-NaN column (it fails while
    binning), and sparse OSM tags such as width_m can be missing at every sensor in a fold.
    The trees never split on a constant column, so the feature is ignored, as older versions
    did, and predicting needs no change.
    """
    empty = np.isnan(x).all(axis=0) if len(x) else np.zeros(x.shape[1], dtype=bool)
    if empty.any():
        x = x.copy()
        x[:, empty] = 0.0
    return model.fit(x, y)


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
