"""Feature utilities for the flow model.

All feature selection, design-matrix construction and column listing live in
src/features/training_frame.py, which is the single authoritative source for
the pipeline. This module re-exports what model training code needs and adds
the calibration helpers that are specific to per-sensor post-processing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Re-export the training-frame API so model code only imports from one place.
from src.features.training_frame import (  # noqa: F401
    Frame,
    derived_features,
    design_matrix,
    feature_names,
    fold_plan,
    load,
    row_features,
    segment_features,
    sensor_features,
)

# ---------------------------------------------------------------------------
# Per-sensor calibration
# ---------------------------------------------------------------------------


def calibration_window(sensors: np.ndarray, ts: pd.Series, days: float) -> np.ndarray:
    """True for each sensor's first `days` days: the counts used to calibrate it.

    Both calibrated and uncalibrated predictions are scored on the remaining rows,
    so the two are always measured on the same hours.
    """
    first = pd.Series(ts.to_numpy()).groupby(sensors).transform("min")
    return (ts.to_numpy() - first.to_numpy()) < pd.Timedelta(days=days)


def calibrate(
    pred: np.ndarray, y: np.ndarray, sensors: np.ndarray, window: np.ndarray
) -> np.ndarray:
    """Scale each sensor's predictions to match its observed counts in the window.

    This is the per-site calibration step (CLAUDE.md "shape vs. scale"): the model
    learns relative patterns; a few local counts set the absolute volume.
    A sensor with zero predicted pedestrians in its window keeps its predictions.
    """
    frame = pd.DataFrame({"s": sensors, "y": np.where(window, y, 0.0)})
    frame["p"] = np.where(window, pred, 0.0)
    sums = frame.groupby("s")[["y", "p"]].sum()
    scale = (sums["y"] / sums["p"]).where(sums["p"] > 0, 1.0)
    return pred * scale.reindex(sensors).to_numpy()
