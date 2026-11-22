"""Footpath slope and elevation for every Melbourne OSM segment, for serving predictions.

Usage: python -m src.features.segment_steepness

The rich flow model takes three features from the footpath-steepness survey (slope_pct,
slope_pct_local_mean, elevation_m). Training reads them per count sensor
(src/features/sensor_steepness.py); the prediction API needs them for any segment. Each
segment gets them the same way, with the same functions and settings
(`sensor_steepness` in configs/features.yaml), at its midpoint:
- a segment with a count sensor on it takes that sensor's values, so served and trained
  features match exactly there (the lowest location_id when several sensors share it);
- otherwise the nearest footpath section within max_match_m and the length-weighted mean
  within local_radius_m (sensor_steepness.match_footpaths);
- otherwise elevation and terrain slope from the GA 5 m LiDAR DEM (fill_from_dem), with no
  local mean, as for sensors; outside the DEM all three stay empty.

`place_source` says which applied: `sensor`, `footpath_steepness` or `ga_dem_5m`.

Writes data/processed/features/melbourne/segment_steepness.parquet, one row per segment.
"""

from __future__ import annotations

import time
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from src.features import sensor_steepness as ss
from src.features.common import (
    config_hash,
    load_features_config,
    log,
    write_meta,
    write_parquet,
)
from src.ingest.common import REPO_ROOT, load_config, setup_logging

COLUMNS = ["slope_pct", "slope_pct_local_mean", "elevation_m"]
SENSOR = "sensor"


def sensor_values(feats) -> pd.DataFrame:
    """The training values per segment that has an outdoor count sensor on it."""
    steep = pd.read_parquet(feats / "sensor_steepness.parquet")
    seg = pd.read_parquet(
        feats / "sensor_segment_map.parquet", columns=["location_id", "segment_id"]
    ).dropna()
    both = steep.merge(seg, on="location_id").sort_values("location_id")
    both = both.drop_duplicates("segment_id")
    out = both[["segment_id", *COLUMNS]].copy()
    out["place_source"] = SENSOR
    out["sensor_location_id"] = both["location_id"].to_numpy()
    return out


def build(
    data_cfg: dict[str, Any], fcfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = fcfg["sensor_steepness"]
    site = cfg["site"]
    crs = fcfg["sites"][site]["crs"]
    raw = REPO_ROOT / data_cfg["raw_dir"]
    feats = REPO_ROOT / fcfg["processed_dir"] / "features" / site
    steepness_path = raw / "melbourne_portal" / "footpath-steepness.geojson"
    dem_path = raw / "dem" / f"{site}_ga_lidar_5m.tif"
    for p in (steepness_path, dem_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is missing; run `dvc pull`, or the pull_static_layers task, first"
            )

    segs = gpd.read_parquet(
        feats / "osm_segment_features.parquet", columns=["segment_id", "geometry"]
    ).to_crs(crs)
    # match_footpaths and fill_from_dem key their rows on location_id; here it is the
    # segment's position in `segs`.
    points = gpd.GeoDataFrame(
        {"location_id": np.arange(len(segs), dtype="int64")},
        geometry=segs.geometry.interpolate(0.5, normalized=True).to_numpy(),
        crs=crs,
    )
    sections, stats = ss.usable_sections(
        gpd.read_file(steepness_path), crs, cfg["min_section_m"]
    )
    table = ss.match_footpaths(
        points, sections, cfg["max_match_m"], cfg["local_radius_m"]
    )
    table = ss.fill_from_dem(table, points, dem_path, cfg["dem_offset_m"])
    table = table.sort_values("location_id").reset_index(drop=True)
    table.insert(0, "segment_id", segs["segment_id"].to_numpy()[table["location_id"]])
    table = table.drop(columns=["location_id", "section_id", "n_local_sections"])
    table["sensor_location_id"] = pd.array([pd.NA] * len(table), dtype="Int64")

    sensors = sensor_values(feats).set_index("segment_id")
    on_sensor = table["segment_id"].isin(sensors.index)
    idx = table.loc[on_sensor, "segment_id"]
    for c in [*COLUMNS, "place_source", "sensor_location_id"]:
        table.loc[on_sensor, c] = sensors.loc[idx, c].to_numpy()
    table.loc[on_sensor, "match_distance_m"] = np.nan

    for c in [*COLUMNS, "match_distance_m"]:
        table[c] = table[c].astype("float32")
    table["place_source"] = table["place_source"].astype("category")

    by_source = table["place_source"].value_counts(dropna=False)
    meta = {
        **stats,
        "segments": len(table),
        "segments_by_source": {str(k): int(v) for k, v in by_source.items()},
        "segments_without_slope": int(table["slope_pct"].isna().sum()),
    }
    return table, meta


def main() -> None:
    setup_logging()
    data_cfg = load_config()
    fcfg = load_features_config()
    t = time.perf_counter()
    table, meta = build(data_cfg, fcfg)
    out = (
        REPO_ROOT
        / fcfg["processed_dir"]
        / "features"
        / fcfg["sensor_steepness"]["site"]
    )
    write_parquet(table, out / "segment_steepness.parquet")
    write_meta(
        out / "segment_steepness_meta.json",
        config_hash=config_hash(fcfg["sensor_steepness"]),
        runtime_s=round(time.perf_counter() - t, 1),
        **meta,
    )
    log.info("segment_steepness: %s", meta)


if __name__ == "__main__":
    main()
