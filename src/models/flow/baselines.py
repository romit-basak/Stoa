"""The baseline ladder: what network structure alone predicts, before any learned model.

Each rung is fitted on the same fold plan as the candidates:
1. betweenness: shortest-path betweenness at 1600 m;
2. space_syntax: every Space Syntax-style metric (metric and angular, three radii).

Space Syntax predicts movement potential, not hourly counts, and needs calibration against
counts (docs/scoping, "Space Syntax ... needs calibration"). So every rung has the same form:
log1p(count) = typical log count for the hour of the week + a ridge regression of the rest on
the rung's network terms, both fitted to the training sensors' counts. Held-out sensors are
predicted from their network terms alone, like the candidates.

The network terms are constant per sensor, so the ridge is fitted on one row per sensor (its
mean log residual), with the penalty chosen by leave-one-out over those sensors. Fitted on
every hourly row instead, a few dozen sensors' terms repeated 200,000 times overfit, and the
richer Space Syntax rung scored worse than betweenness alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

RUNGS = ("betweenness", "space_syntax")
ALPHAS = np.logspace(-3, 3, 13)


def network_columns(rung: str, osm: pd.DataFrame) -> list[str]:
    if rung == "betweenness":
        return ["met_betweenness_1600"]
    return [c for c in osm.columns if c.startswith(("ang_", "met_", "junctions_"))]


class NetworkBaseline:
    """Hour-of-week profile in log counts plus a ridge on per-sensor network terms.

    alpha None: the penalty is chosen by leave-one-out over the training sensors.
    """

    def __init__(self, columns: list[str], alpha: float | None = None):
        self.columns = columns
        self.alpha = alpha

    def _terms(self, osm: pd.DataFrame, sensors: np.ndarray) -> np.ndarray:
        # Betweenness and reach span orders of magnitude, so they enter as log1p.
        v = osm[self.columns].reindex(sensors).to_numpy(dtype="float64")
        return np.sign(v) * np.log1p(np.abs(v))

    def fit(
        self, hourly: pd.DataFrame, osm: pd.DataFrame, rows: np.ndarray, y: np.ndarray
    ) -> NetworkBaseline:
        how = hourly["hour_of_week"].to_numpy()[rows]
        ly = np.log1p(y[rows])
        prof = pd.Series(ly).groupby(how).mean()
        self.profile_ = prof.reindex(range(168)).fillna(float(ly.mean())).to_numpy()
        sensors = hourly["location_id"].to_numpy()[rows]
        resid = pd.Series(ly - self.profile_[how]).groupby(sensors).mean()
        x = self._terms(osm, resid.index.to_numpy())
        self.fill_ = np.nan_to_num(np.nanmean(x, axis=0))
        x = np.where(np.isnan(x), self.fill_, x)
        self.scaler_ = StandardScaler().fit(x)
        ridge = (
            RidgeCV(alphas=ALPHAS) if self.alpha is None else Ridge(alpha=self.alpha)
        )
        self.ridge_ = ridge.fit(self.scaler_.transform(x), resid.to_numpy())
        return self

    def predict(
        self, hourly: pd.DataFrame, osm: pd.DataFrame, rows: np.ndarray
    ) -> np.ndarray:
        how = hourly["hour_of_week"].to_numpy()[rows]
        x = self._terms(osm, hourly["location_id"].to_numpy()[rows])
        x = np.where(np.isnan(x), self.fill_, x)
        log_pred = self.profile_[how] + self.ridge_.predict(self.scaler_.transform(x))
        return np.maximum(np.expm1(log_pred), 0.0).astype("float32")


def oof(
    rung: str,
    hourly: pd.DataFrame,
    osm: pd.DataFrame,
    y: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Out-of-fold predictions of one rung (NaN on rows in no test fold)."""
    cols = network_columns(rung, osm)
    pred = np.full(len(hourly), np.nan, dtype="float32")
    for train, test in plan:
        model = NetworkBaseline(cols).fit(hourly, osm, train, y)
        pred[test] = model.predict(hourly, osm, test)
    return pred
