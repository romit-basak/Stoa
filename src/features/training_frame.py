"""The training frame for the Melbourne flow model: rows, features and location splits.

Usage: python -m src.features.training_frame [--drop-needs-review]

Every model trained on the counts builds its inputs here, so the leak traps live in one tested
place instead of in each training script:
- rows: located outdoor sensors, hours not flagged as outages, and the training site only.
  NYC and Boston counts can never enter (CLAUDE.md: NYC is a transfer test, never training);
- columns: chosen by role in src/features/schema.py. IDs, coordinates, the target and the
  direction counts that add up to it are never features (core rules 2 and 9). `rich` uses every
  feature; `portable` only those any city has;
- the Metro Tunnel: the OSM snapshot postdates the opening (2025-11-30), so before it the new
  stations' distance is set to the cap, as if they weren't there;
- splits by location (core rule 2): sensors all within `group_radius_m` of each other share a
  group, so close neighbours on one street never straddle train and test; the within-Melbourne
  holdout precincts (configs/bias.yaml) are in no fold's training set.

The frame itself (7.4M rows x ~140 columns, about 4 GB as float32) is not written. `X()` builds
it in row chunks on demand. Written to data/processed/training/<site>/:
- splits.parquet: one row per sensor: group, fold (-1 for holdout sensors), holdout precinct;
- event_days.parquet: dates with an AFL game or a logged event at a venue, to report apart;
- frame_spec.json: the columns, input hashes, the DVC data version, the OSM extract, reference
  quantiles of every feature (the baseline for drift checks) and the needs_review count.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.model_selection import GroupKFold

from src.features import schema
from src.features.common import (
    config_hash,
    load_features_config,
    log,
    site_settings,
    write_meta,
    write_parquet,
)
from src.features.sensor_match import load_sensor_locations
from src.ingest.common import REPO_ROOT, load_config, setup_logging, sha256

VARIANTS = ("rich", "portable")
QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
# A column with at most this many distinct values is compared by category, not by bins.
MAX_CATEGORIES = 30


# --------------------------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------------------------


def row_features(variant: str = "rich") -> list[str]:
    """Features of the hourly table itself, in table order."""
    _check_variant(variant)
    return schema.features(schema.tables()["counts_hourly"], variant == "portable")


def segment_features(variant: str = "rich") -> list[str]:
    """Features of each sensor's matched segment, in sensor_osm_features order."""
    _check_variant(variant)
    return schema.features(
        schema.tables()["sensor_osm_features"], variant == "portable"
    )


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}: one of {VARIANTS}")


def sensor_features(osm: pd.DataFrame, variant: str = "rich") -> pd.DataFrame:
    """Segment features per sensor as float32, indexed by location_id."""
    wanted = set(segment_features(variant))
    cols = [c for c in osm.columns if c in wanted]
    return osm.set_index("location_id")[cols].astype("float32")


def design_matrix(
    hourly: pd.DataFrame,
    index: np.ndarray,
    osm: pd.DataFrame,
    station_cap_m: float,
    row_cols: list[str] | None = None,
) -> np.ndarray:
    """Feature matrix for hourly rows `index`: row features, then the sensor's segment features.

    `osm` comes from sensor_features(). Before the Metro Tunnel opened, the new stations'
    distance is set to the cap, as if they weren't there (docs/features/counts_hourly.md).
    """
    row_cols = row_features() if row_cols is None else row_cols
    part = hourly.iloc[index]
    rows = part[row_cols].astype("float32").to_numpy()
    seg = osm.reindex(part["location_id"].to_numpy())
    if "recent_station_dist_m" in seg.columns:
        before = ~part["post_metro_tunnel"].to_numpy(dtype=bool)
        seg.loc[before, "recent_station_dist_m"] = station_cap_m
    return np.hstack([rows, seg.to_numpy(dtype="float32")])


def feature_names(osm: pd.DataFrame, row_cols: list[str] | None = None) -> list[str]:
    return (row_features() if row_cols is None else row_cols) + list(osm.columns)


# --------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------


def check_site(site: str, data_cfg: dict[str, Any], fcfg: dict[str, Any]) -> None:
    """Only the site the hourly table is built for, and one with count sensors, may train."""
    if site != fcfg["counts_hourly"]["site"] or not data_cfg["sites"][site].get(
        "count_sensors"
    ):
        raise ValueError(
            f"{site} is not a training site: only {fcfg['counts_hourly']['site']}'s counts "
            "train models (other sites are transfer tests or demonstrations)"
        )


def eligible(hourly: pd.DataFrame) -> np.ndarray:
    """Positions of rows a model may train or be scored on."""
    ok = (
        hourly["has_location"].to_numpy(dtype=bool)
        & hourly["location_type"].eq("Outdoor").to_numpy()
        & ~hourly["suspect_outage_zero"].to_numpy(dtype=bool)
    )
    return np.flatnonzero(ok)


@dataclass
class Frame:
    """Eligible hourly rows plus per-sensor segment features; X() builds the matrix."""

    hourly: pd.DataFrame
    osm: pd.DataFrame
    row_cols: list[str]
    station_cap_m: float

    @property
    def columns(self) -> list[str]:
        return feature_names(self.osm, self.row_cols)

    @property
    def y(self) -> np.ndarray:
        return self.hourly["pedestriancount"].to_numpy(dtype=float)

    @property
    def sensors(self) -> np.ndarray:
        return self.hourly["location_id"].to_numpy()

    def X(self, index: np.ndarray | None = None, chunk: int = 500_000) -> np.ndarray:
        """Rows `index` (default all) of the feature matrix, built in chunks."""
        idx = np.arange(len(self.hourly)) if index is None else np.asarray(index)
        parts = [
            design_matrix(
                self.hourly,
                idx[i : i + chunk],
                self.osm,
                self.station_cap_m,
                self.row_cols,
            )
            for i in range(0, len(idx), chunk)
        ]
        return (
            np.vstack(parts) if parts else np.empty((0, len(self.columns)), "float32")
        )


def load(
    site: str | None = None,
    variant: str = "rich",
    drop_needs_review: bool = False,
    extra_columns: list[str] | None = None,
) -> Frame:
    data_cfg, fcfg = load_config(), load_features_config()
    site = site or fcfg["counts_hourly"]["site"]
    check_site(site, data_cfg, fcfg)
    feats = REPO_ROOT / fcfg["processed_dir"] / "features" / site
    row_cols = row_features(variant)
    keep = ["location_id", "ts_utc", "date_local", "pedestriancount", "has_location"]
    keep += [
        "location_type",
        "suspect_outage_zero",
        "needs_review",
        "post_metro_tunnel",
    ]
    columns = list(dict.fromkeys(keep + row_cols + (extra_columns or [])))
    hourly = pd.read_parquet(feats / "counts_hourly.parquet", columns=columns)
    rows = eligible(hourly)
    if drop_needs_review:
        rows = rows[
            ~hourly["needs_review"].iloc[rows].fillna(True).to_numpy(dtype=bool)
        ]
    hourly = hourly.iloc[rows].reset_index(drop=True)
    osm = sensor_features(
        pd.read_parquet(feats / "sensor_osm_features.parquet"), variant
    )
    return Frame(hourly, osm, row_cols, fcfg["built_form"]["max_distance_m"])


# --------------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------------


def fold_plan(
    groups: np.ndarray, n_folds: int, train_rows: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(training sample, held-out rows) per fold, as positions into the rows.

    Folds split by group. The training sample is drawn once per fold, so every model compared
    on a plan trains on the same rows.
    """
    rng = np.random.default_rng(seed)
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    plan = []
    for train, test in folds.split(groups, groups=groups):
        if len(train) > train_rows:
            train = np.sort(rng.choice(train, train_rows, replace=False))
        plan.append((train, test))
    return plan


def sensor_groups(xy: np.ndarray, radius_m: float) -> np.ndarray:
    """Group id per sensor: clusters whose members are all within radius_m of each other.

    Complete linkage, so close neighbours share a group but a chain of sensors along a dense
    street doesn't grow into one group covering the whole CBD (which single linkage does).
    """
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=int)
    tree = linkage(pdist(xy), method="complete")
    return fcluster(tree, t=radius_m, criterion="distance") - 1


def splits(
    sensors: pd.DataFrame,
    holdouts: dict[str, list[int]],
    n_folds: int,
    seed: int,
    radius_m: float,
) -> pd.DataFrame:
    """One row per sensor: its spatial group, fold, and holdout precinct (if any).

    `sensors` has location_id, x, y in metres. Holdout sensors get fold -1, and a group that
    touches a holdout sensor is held out whole, so no neighbour of a holdout trains.
    """
    s = sensors[["location_id", "x", "y"]].reset_index(drop=True).copy()
    s["group"] = sensor_groups(s[["x", "y"]].to_numpy(), radius_m)
    precinct = {lid: name for name, ids in holdouts.items() for lid in ids}
    s["holdout"] = s["location_id"].map(precinct).fillna("")
    held_groups = set(s.loc[s["holdout"] != "", "group"])
    s.loc[s["group"].isin(held_groups) & (s["holdout"] == ""), "holdout"] = "neighbour"
    s["fold"] = -1
    train = s["holdout"] == ""
    folds = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(folds.split(s[train], groups=s.loc[train, "group"])):
        s.loc[s.index[train][test], "fold"] = k
    return s[["location_id", "group", "fold", "holdout"]]


def event_days(
    afl: pd.DataFrame, afl_venues: dict[str, str], events: dict[str, Any]
) -> pd.DataFrame:
    """Local dates with an AFL game at the listed venues or a logged event at a venue."""
    a = afl[afl["venue"].isin(afl_venues)]
    rows = [
        {"date_local": pd.Timestamp(d), "kind": "afl", "venue": afl_venues[v]}
        for d, v in zip(pd.to_datetime(a["date"]), a["venue"])
    ]
    for e in events.get("events", []):
        if e.get("venue"):
            day = pd.Timestamp(e["start"]).normalize()
            rows.append({"date_local": day, "kind": e["kind"], "venue": e["venue"]})
    if not rows:
        return pd.DataFrame(columns=["date_local", "kinds", "venues"])
    df = pd.DataFrame(rows)
    return (
        df.groupby("date_local")
        .agg(
            kinds=("kind", lambda k: ",".join(sorted(set(k)))),
            venues=("venue", lambda v: ",".join(sorted(set(v)))),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------------------------


def reference_quantiles(values: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per column: quantiles, missing share and a distribution later drift checks compare to.

    The distribution is ten bins at the deciles for a numeric column, or each value's share for
    one with few distinct values (flags, hours, weekdays), where deciles would tie.
    """
    out = {}
    for c in values.columns:
        v = pd.to_numeric(values[c].astype("float64"), errors="coerce")
        q = v.quantile(QUANTILES)
        ref: dict[str, Any] = {
            "quantiles": {
                str(k): (None if pd.isna(x) else float(x)) for k, x in q.items()
            },
            "null_share": float(v.isna().mean()),
        }
        present = v.dropna()
        if present.nunique() <= MAX_CATEGORIES:
            shares = present.value_counts(normalize=True).sort_index()
            ref["categories"] = {str(float(k)): float(s) for k, s in shares.items()}
        elif len(present):
            edges = np.unique(present.quantile(np.linspace(0, 1, 11)).to_numpy())
            ref["edges"] = [float(x) for x in edges]
            ref["shares"] = [float(s) for s in bin_shares(present, edges)]
        out[c] = ref
    return out


def bin_shares(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Share of values in each bin between edges; values outside go to the end bins."""
    inner = np.asarray(edges, dtype=float)[1:-1]
    idx = np.searchsorted(inner, values.to_numpy(dtype=float), side="right")
    counts = np.bincount(idx, minlength=len(edges) - 1)
    return counts / max(counts.sum(), 1)


def psi(expected: np.ndarray, actual: np.ndarray, floor: float = 1e-4) -> float:
    """Population stability index between two sets of bin shares."""
    e = np.maximum(np.asarray(expected, dtype=float), floor)
    a = np.maximum(np.asarray(actual, dtype=float), floor)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_from_reference(values: pd.Series, ref: dict[str, Any]) -> float | None:
    """PSI of a column against its reference distribution (None when there is none)."""
    v = pd.to_numeric(values.astype("float64"), errors="coerce").dropna()
    if not len(v):
        return None
    if "categories" in ref:
        cats = [float(k) for k in ref["categories"]]
        expected = np.array(list(ref["categories"].values()))
        shares = v.value_counts(normalize=True)
        actual = np.array([shares.get(k, 0.0) for k in cats])
        # Values the reference never saw count against the drift too.
        unseen = 1.0 - actual.sum()
        if unseen > 0:
            expected, actual = np.append(expected, 0.0), np.append(actual, unseen)
        return psi(expected, actual)
    if "edges" in ref:
        return psi(np.array(ref["shares"]), bin_shares(v, np.array(ref["edges"])))
    return None


def dvc_version(pointer: Path) -> str | None:
    if not pointer.exists():
        return None
    return yaml.safe_load(pointer.read_text())["outs"][0]["md5"]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-needs-review", action="store_true")
    args = ap.parse_args()
    t = time.perf_counter()
    data_cfg, fcfg = load_config(), load_features_config()
    from src.bias.slices import load_bias_config

    tcfg, bias = fcfg["training_frame"], load_bias_config()
    site = site_settings(fcfg["counts_hourly"]["site"], data_cfg, fcfg)
    frame = load(site.name, "rich", args.drop_needs_review)

    locs = load_sensor_locations(data_cfg).to_crs(site.crs)
    used = locs[locs["location_id"].isin(np.unique(frame.sensors))]
    used = used.drop_duplicates("location_id")
    xy = pd.DataFrame(
        {
            "location_id": used["location_id"].astype(int),
            "x": used.geometry.x,
            "y": used.geometry.y,
        }
    )
    split = splits(
        xy,
        bias["holdout_precincts"],
        tcfg["folds"],
        tcfg["seed"],
        tcfg["group_radius_m"],
    )

    raw = REPO_ROOT / data_cfg["raw_dir"]
    afl_path = raw / "events" / "afl_attendance.csv"
    afl = (
        pd.read_csv(afl_path)
        if afl_path.exists()
        else pd.DataFrame(columns=["date", "venue"])
    )
    events = yaml.safe_load((REPO_ROOT / "configs" / "events.yaml").read_text())
    days = event_days(afl, bias["afl_venues"], events)

    out = REPO_ROOT / fcfg["processed_dir"] / "training" / site.name
    write_parquet(split, out / "splits.parquet")
    write_parquet(days, out / "event_days.parquet")

    feats = REPO_ROOT / fcfg["processed_dir"] / "features" / site.name
    rows_ref = reference_quantiles(frame.hourly[frame.row_cols])
    seg_ref = reference_quantiles(frame.osm.loc[np.unique(frame.sensors)])
    extract = json.loads((site.interim / "extract_meta.json").read_text())
    needs_review = frame.hourly.drop_duplicates("location_id")["needs_review"]
    write_meta(
        out / "frame_spec.json",
        site=site.name,
        config_hash=config_hash(tcfg),
        drop_needs_review=args.drop_needs_review,
        rows=len(frame.hourly),
        sensors=len(np.unique(frame.sensors)),
        sensors_needs_review=int(needs_review.fillna(True).sum()),
        features={v: row_features(v) + segment_features(v) for v in VARIANTS},
        holdout_sensors=int((split["fold"] == -1).sum()),
        folds=tcfg["folds"],
        event_days=len(days),
        inputs={
            name: sha256(feats / name)
            for name in ["counts_hourly.parquet", "sensor_osm_features.parquet"]
        },
        dvc_data_version=dvc_version(REPO_ROOT / "data" / "processed.dvc"),
        osm_pbf_sha256=extract["pbf_sha256"],
        reference={"rows": rows_ref, "segments": seg_ref},
        runtime_s=round(time.perf_counter() - t, 1),
    )
    log.info(
        "training frame: %d rows, %d sensors (%d held out), %d event days",
        len(frame.hourly),
        len(np.unique(frame.sensors)),
        int((split["fold"] == -1).sum()),
        len(days),
    )


if __name__ == "__main__":
    main()
