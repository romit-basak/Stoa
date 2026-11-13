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

# Each family's minimum leaf size. Configured and tuned values are per train_rows_per_fold rows
# (the location folds' training sample), so that a leaf has to span several sensors whatever
# the sample size; `leaf_scaled` converts them for fits on other sample sizes.
LEAF_PARAM = {
    "hgb": "min_samples_leaf",
    "lgbm": "min_child_samples",
    "xgb": "min_child_weight",
}


def leaf_scaled(family: str, params: dict[str, Any], factor: float) -> dict[str, Any]:
    """params with the minimum leaf size multiplied by `factor` (fit rows / fold rows)."""
    name = LEAF_PARAM.get(family)
    if name is None or name not in params or factor == 1:
        return dict(params)
    value = params[name] * factor
    if family == "xgb":
        return {**params, name: max(1e-3, float(value))}
    return {**params, name: max(1, round(value))}


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


def fit(
    model: Any, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
) -> Any:
    """Fit `model`, with any feature that has no values in these rows set to a constant.

    scikit-learn 1.9 crashes HistGradientBoosting on an all-NaN column (it fails while
    binning), and sparse OSM tags such as width_m can be missing at every sensor in a fold.
    The trees never split on a constant column, so the feature is ignored, as older versions
    did, and predicting needs no change. `sample_weight` weights the rows (bias mitigation).
    """
    empty = np.isnan(x).all(axis=0) if len(x) else np.zeros(x.shape[1], dtype=bool)
    if empty.any():
        x = x.copy()
        x[:, empty] = 0.0
    if sample_weight is None:
        return model.fit(x, y)
    return model.fit(x, y, sample_weight=sample_weight)


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
