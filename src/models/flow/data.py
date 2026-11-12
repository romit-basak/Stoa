"""The flow model's inputs: the training frame with its folds, and the calibration helpers.

Rows and features come from src/features/training_frame.py; folds come from its
splits.parquet, which groups neighbouring sensors and puts the holdout precincts in no fold
(fold -1). Every candidate and baseline is trained and scored on the same fold plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bias.slice_errors import calibrate, calibration_window
from src.features import training_frame as tf
from src.features.common import load_features_config
from src.ingest.common import REPO_ROOT
from src.models.tracking import load_models_config

__all__ = [
    "Data",
    "calibrate",
    "calibration_window",
    "final_rows",
    "flow_config",
    "fold_plan",
    "load",
    "output_dir",
]


def flow_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return (cfg or load_models_config())["flow"]


def output_dir(fcfg: dict[str, Any], override: str | Path | None = None) -> Path:
    path = Path(override or fcfg["output_dir"])
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class Data:
    """A training frame with each row's fold (-1: holdout or its neighbour), spatial group
    and holdout precinct."""

    frame: tf.Frame
    fold: np.ndarray
    group: np.ndarray
    holdout: np.ndarray
    site: str
    variant: str

    @property
    def y(self) -> np.ndarray:
        return self.frame.y

    @property
    def sensors(self) -> np.ndarray:
        return self.frame.sensors

    @property
    def ts(self) -> pd.Series:
        return self.frame.hourly["ts_utc"]


def load(variant: str, site: str | None = None) -> Data:
    site = site or load_features_config()["counts_hourly"]["site"]
    frame = tf.load(site, variant, extra_columns=["hour_of_week"])
    path = (
        REPO_ROOT
        / load_features_config()["processed_dir"]
        / "training"
        / site
        / "splits.parquet"
    )
    splits = pd.read_parquet(path).set_index("location_id")
    fold = splits["fold"].reindex(frame.sensors)
    if fold.isna().any():
        missing = sorted(set(frame.sensors[fold.isna().to_numpy()]))
        raise ValueError(
            f"sensors {missing} have no split; rebuild it with "
            "`python -m src.features.training_frame`"
        )
    group = splits["group"].reindex(frame.sensors).to_numpy(dtype=int)
    holdout = splits["holdout"].reindex(frame.sensors).fillna("").to_numpy(object)
    return Data(frame, fold.to_numpy(dtype=int), group, holdout, site, variant)


def fold_plan(
    fold: np.ndarray, train_rows: int, seed: int, n_folds: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(training sample, test rows) per fold, as row positions.

    Training rows are every other fold's, never a holdout's, sampled down to train_rows. The
    sample is drawn once per fold, so every model compared on the plan trains on the same rows.
    n_folds keeps only the first folds (quick runs); the training sample still uses all others.
    """
    rng = np.random.default_rng(seed)
    ks = sorted(int(k) for k in np.unique(fold) if k >= 0)[:n_folds]
    plan = []
    for k in ks:
        test = np.flatnonzero(fold == k)
        train = np.flatnonzero((fold >= 0) & (fold != k))
        if len(train) > train_rows:
            train = np.sort(rng.choice(train, train_rows, replace=False))
        plan.append((train, test))
    return plan


def final_rows(fold: np.ndarray, train_rows: int, seed: int) -> np.ndarray:
    """Rows for the final fit: every sensor in a fold (no holdout), sampled to train_rows."""
    rows = np.flatnonzero(fold >= 0)
    if len(rows) > train_rows:
        rows = np.sort(
            np.random.default_rng(seed).choice(rows, train_rows, replace=False)
        )
    return rows
