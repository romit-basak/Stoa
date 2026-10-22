"""Footpath slope and elevation for each Melbourne count sensor.

Usage: python -m src.features.sensor_steepness

Writes data/processed/features/melbourne/sensor_steepness.parquet, one row per outdoor sensor,
which joins onto the counts on location_id (counts_hourly does this).

Based on Shubham Hadawle's Colab join notebook. Each sensor is matched to the nearest section of
the City of Melbourne footpath-steepness survey, which gives its slope (`gradepc`) and elevation
(the mean of `rlmin` and `rlmax`, in metres AHD). Fixes over the notebook:
- sections with no grade are skipped, so a sensor is never matched to an empty section when a
  measured one is close by;
- sections shorter than min_section_m are skipped: their grades run up to 580%;
- a length-weighted mean over all sections within local_radius_m is added, so one odd section
  doesn't decide a sensor's slope;
- sensors with no section within max_match_m take elevation and terrain slope from the GA 5 m
  LiDAR DEM instead of being left empty. That slope is the steepest terrain gradient, not the
  grade along a footpath, so `place_source` says which method each row used.

The 18 archive sensors missing from today's sensor-locations file get no row: they have no
coordinates yet (see CLAUDE.md, "Open decisions").
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from src.features.common import (
    config_hash,
    load_features_config,
    log,
    write_meta,
    write_parquet,
)
from src.features.sensor_match import sensor_locations_path
from src.ingest.common import REPO_ROOT, load_config, setup_logging

FOOTPATH = "footpath_steepness"
DEM = "ga_dem_5m"


def outdoor_sensors(path: Path, crs: int) -> gpd.GeoDataFrame:
    """Outdoor sensors with coordinates, one row each, in the site's metric CRS."""
    s = gpd.read_file(path)
    s = s[s["location_type"].eq("Outdoor") & s.geometry.notna()]
    s = s.drop_duplicates("location_id").copy()
    s["location_id"] = s["location_id"].astype("int64")
    return s[["location_id", "sensor_description", "geometry"]].to_crs(crs)


def usable_sections(
    steepness: gpd.GeoDataFrame, crs: int, min_section_m: float
) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Sections with a measured grade and elevation, at least min_section_m long."""
    s = steepness.to_crs(crs)
    no_grade = s[["gradepc", "rlmin", "rlmax"]].isna().any(axis=1)
    short = ~no_grade & (s["distance"] < min_section_m)
    kept = s[~no_grade & ~short].copy()
    kept["elevation_m"] = (kept["rlmin"] + kept["rlmax"]) / 2
    kept = kept[["mcc_id", "gradepc", "distance", "elevation_m", "geometry"]]
    stats = {
        "sections": len(s),
        "sections_no_grade": int(no_grade.sum()),
        "sections_short": int(short.sum()),
        "sections_used": len(kept),
    }
    return kept.reset_index(drop=True), stats


def match_footpaths(
    sensors: gpd.GeoDataFrame,
    sections: gpd.GeoDataFrame,
    max_match_m: float,
    local_radius_m: float,
) -> pd.DataFrame:
    """Nearest section per sensor (within max_match_m), plus a length-weighted local mean."""
    near = gpd.sjoin_nearest(
        sensors,
        sections,
        how="left",
        max_distance=max_match_m,
        distance_col="match_distance_m",
    )
    # Equidistant sections (a sensor on a shared edge) would duplicate the sensor; keep one,
    # breaking ties on the section ID so reruns pick the same one.
    near = near.sort_values(["location_id", "match_distance_m", "mcc_id"])
    near = near.drop_duplicates("location_id")
    out = pd.DataFrame(
        {
            "location_id": near["location_id"].to_numpy(),
            "slope_pct": near["gradepc"].to_numpy(dtype=float),
            "elevation_m": near["elevation_m"].to_numpy(dtype=float),
            "match_distance_m": near["match_distance_m"].to_numpy(dtype=float),
            "section_id": near["mcc_id"].astype("Int64").to_numpy(),
        }
    )

    ring = sensors.set_geometry(sensors.buffer(local_radius_m))
    local = gpd.sjoin(ring, sections, how="inner", predicate="intersects")
    local["w"] = local["gradepc"] * local["distance"]
    agg = local.groupby("location_id").agg(
        w=("w", "sum"), length=("distance", "sum"), n=("gradepc", "size")
    )
    agg["slope_pct_local_mean"] = agg["w"] / agg["length"]
    out = out.merge(
        agg[["slope_pct_local_mean", "n"]].rename(columns={"n": "n_local_sections"}),
        on="location_id",
        how="left",
    )
    out["n_local_sections"] = out["n_local_sections"].fillna(0).astype("int64")
    out["place_source"] = np.where(out["slope_pct"].notna(), FOOTPATH, None)
    return out


def dem_terrain(
    points: gpd.GeoSeries, dem_path: Path, offset_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Elevation at each point and the terrain slope (%) from samples offset_m to each side.

    Slope is the gradient magnitude from central differences east-west and north-south.
    `points` must be in a metric CRS.
    """
    xs, ys = points.x.to_numpy(), points.y.to_numpy()
    offsets = [(0, 0), (offset_m, 0), (-offset_m, 0), (0, offset_m), (0, -offset_m)]
    with rasterio.open(dem_path) as dem:
        to_dem = Transformer.from_crs(points.crs, dem.crs, always_xy=True)
        z = []
        for dx, dy in offsets:
            lon, lat = to_dem.transform(xs + dx, ys + dy)
            vals = np.array([v[0] for v in dem.sample(zip(lon, lat))], dtype=float)
            if dem.nodata is not None:
                vals[vals == dem.nodata] = np.nan
            z.append(vals)
    centre, east, west, north, south = z
    dzdx = (east - west) / (2 * offset_m)
    dzdy = (north - south) / (2 * offset_m)
    return centre, 100 * np.hypot(dzdx, dzdy)


def fill_from_dem(
    table: pd.DataFrame,
    sensors: gpd.GeoDataFrame,
    dem_path: Path,
    offset_m: float,
) -> pd.DataFrame:
    """Elevation and terrain slope from the DEM for sensors no footpath section matched."""
    missing = table["slope_pct"].isna()
    if not missing.any():
        return table
    pts = sensors.set_index("location_id").loc[table.loc[missing, "location_id"]]
    elevation, slope = dem_terrain(pts.geometry, dem_path, offset_m)
    table = table.copy()
    table.loc[missing, "elevation_m"] = elevation
    table.loc[missing, "slope_pct"] = slope
    table.loc[missing, "place_source"] = np.where(np.isnan(slope), None, DEM)
    return table


def build(
    data_cfg: dict[str, Any], fcfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = fcfg["sensor_steepness"]
    crs = fcfg["sites"][cfg["site"]]["crs"]
    raw = REPO_ROOT / data_cfg["raw_dir"]
    steepness_path = raw / "melbourne_portal" / "footpath-steepness.geojson"
    dem_path = raw / "dem" / f"{cfg['site']}_ga_lidar_5m.tif"
    for p in (steepness_path, dem_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is missing; run `dvc pull`, or the pull_terrain task, first"
            )

    sensors = outdoor_sensors(sensor_locations_path(data_cfg), crs)
    sections, stats = usable_sections(
        gpd.read_file(steepness_path), crs, cfg["min_section_m"]
    )
    table = match_footpaths(
        sensors, sections, cfg["max_match_m"], cfg["local_radius_m"]
    )
    table = fill_from_dem(table, sensors, dem_path, cfg["dem_offset_m"])
    table = sensors.drop(columns="geometry").merge(table, on="location_id")
    table["place_source"] = table["place_source"].astype("category")

    by_source = table["place_source"].value_counts(dropna=False)
    meta = {
        **stats,
        "sensors": len(table),
        "sensors_by_source": {str(k): int(v) for k, v in by_source.items()},
        "sensors_from_dem": table.loc[
            table["place_source"].eq(DEM), "location_id"
        ].tolist(),
        "sensors_without_slope": table.loc[
            table["slope_pct"].isna(), "location_id"
        ].tolist(),
        "sensors_slope_over_10pct": table.loc[
            table["slope_pct"] > 10, ["location_id", "slope_pct"]
        ]
        .round(2)
        .values.tolist(),
        "max_match_distance_m": round(float(table["match_distance_m"].max()), 2),
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
    write_parquet(table, out / "sensor_steepness.parquet")
    write_meta(
        out / "sensor_steepness_meta.json",
        config_hash=config_hash(fcfg["sensor_steepness"]),
        runtime_s=round(time.perf_counter() - t, 1),
        **meta,
    )
    log.info("sensor_steepness: %s", meta)


if __name__ == "__main__":
    main()
