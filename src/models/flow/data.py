"""The flow model's inputs: the training frame with its folds, and the calibration helpers.

Rows and features come from src/features/training_frame.py; folds come from its
splits.parquet, which groups neighbouring sensors and puts the holdout precincts in no fold
(fold -1). Every candidate and baseline is trained and scored on the same fold plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src.bias.slice_errors import calibrate, calibration_window
from src.features import training_frame as tf
from src.features.common import load_features_config
from src.ingest.common import REPO_ROOT, load_config
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
    "summary_plan",
    "trainable",
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
    # Fold -> the other folds' sensors within buffer_m of one of its sensors (fold_plan).
    near: dict[int, np.ndarray] = field(default_factory=dict)
    buffer_m: float = 0.0

    @property
    def y(self) -> np.ndarray:
        return self.frame.y

    @property
    def sensors(self) -> np.ndarray:
        return self.frame.sensors

    @property
    def ts(self) -> pd.Series:
        return self.frame.hourly["ts_utc"]


def neighbours_of_folds(
    xy: pd.DataFrame, fold: pd.Series, radius_m: float
) -> dict[int, np.ndarray]:
    """For each fold, the other folds' sensors closer than radius_m to one of its sensors.

    The spatial groups keep each group within radius_m across, so a sensor just outside a
    group can sit 105 m from one of its members and land in another fold. fold_plan leaves
    these sensors out of the fold's training rows. `xy` (x, y in metres) and `fold` are
    indexed by location_id; holdout sensors (fold -1) are never in a fold's training rows anyway.
    """
    ids = xy.index.to_numpy()
    f = fold.reindex(ids).to_numpy()
    d = squareform(pdist(xy[["x", "y"]].to_numpy()))
    out = {}
    for k in sorted(int(v) for v in np.unique(f) if v >= 0):
        close = (d[f == k] < radius_m).any(axis=0) & (f >= 0) & (f != k)
        out[k] = ids[close]
    return out


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
    fcfg = load_features_config()
    buffer_m = fcfg["training_frame"]["group_radius_m"]
    xy = tf.sensor_xy(frame.sensors, load_config(), fcfg["sites"][site]["crs"])
    near = neighbours_of_folds(xy.set_index("location_id"), splits["fold"], buffer_m)
    return Data(
        frame, fold.to_numpy(dtype=int), group, holdout, site, variant, near, buffer_m
    )


def trainable(data: Data, start: str | None) -> np.ndarray | None:
    """Rows a model may train on when training starts at `start` (a date, Melbourne time),
    or None for every row. Only training rows are limited: every model still predicts all of
    its fold's rows, so models trained from different starts are scored on the same hours."""
    if not start:
        return None
    ts = data.ts
    when = pd.Timestamp(start).tz_localize("Australia/Melbourne").tz_convert("UTC")
    if ts.dt.tz is None:  # naive timestamps are UTC
        when = when.tz_localize(None)
    return (ts >= when).to_numpy()


def fold_plan(
    fold: np.ndarray,
    train_rows: int,
    seed: int,
    n_folds: int | None = None,
    allowed: np.ndarray | None = None,
    sensors: np.ndarray | None = None,
    near: dict[int, np.ndarray] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(training sample, test rows) per fold, as row positions.

    Training rows are every other fold's, never a holdout's, sampled down to train_rows. The
    sample is drawn once per fold, so every model compared on the plan trains on the same rows.
    n_folds keeps only the first folds (quick runs); the training sample still uses all others.
    `allowed` (from `trainable`) limits the training rows; the test rows are always the whole
    fold. With `near` (Data.near) and `sensors` (each row's location_id), sensors close to a
    fold's test sensors are left out of its training rows, so no model trains on a test
    sensor's neighbour (CLAUDE.md core rule 2).
    """
    rng = np.random.default_rng(seed)
    ks = sorted(int(k) for k in np.unique(fold) if k >= 0)[:n_folds]
    usable = (fold >= 0) if allowed is None else (fold >= 0) & allowed
    plan = []
    for k in ks:
        test = np.flatnonzero(fold == k)
        keep = usable & (fold != k)
        if near:
            keep &= ~np.isin(sensors, near.get(k, []))
        train = np.flatnonzero(keep)
        if len(train) > train_rows:
            train = np.sort(rng.choice(train, train_rows, replace=False))
        plan.append((train, test))
    return plan


def final_rows(
    fold: np.ndarray, train_rows: int, seed: int, allowed: np.ndarray | None = None
) -> np.ndarray:
    """Rows for the final fit: every sensor in a fold (no holdout), sampled to train_rows,
    limited to `allowed` rows if given."""
    rows = np.flatnonzero((fold >= 0) if allowed is None else (fold >= 0) & allowed)
    if len(rows) > train_rows:
        rows = np.sort(
            np.random.default_rng(seed).choice(rows, train_rows, replace=False)
        )
    return rows


def summary_plan(
    data: Data, summary: dict[str, Any]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The fold plan a train run used, rebuilt from its candidates.json (runs from before the
    neighbour buffer have no buffer_m and are rebuilt without it)."""
    buffered = bool(summary.get("buffer_m"))
    return fold_plan(
        data.fold,
        summary["train_rows"],
        summary["seed"],
        summary["n_folds"],
        trainable(data, summary.get("train_start")),
        data.sensors if buffered else None,
        data.near if buffered else None,
    )
