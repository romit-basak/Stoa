"""Representation report: how much data each slice of the hourly table has (bias step 1).

Usage: python -m src.bias.representation

Writes to data/processed/bias/melbourne/:
- slice_representation.csv: one row per (dimension, group);
- slice_crosstabs.csv: the same for pairs of slices (configs/bias.yaml, crosstabs);
- coverage_by_year.csv: rows, sensors and pedestrians per local year;
- sensor_slices.csv: one row per sensor with its slices and the values behind them;
- slice_representation_meta.json.
It needs no model: it shows which places, times and conditions the training data
under-represents, before any error is measured. The count distribution per group (median,
90th percentile, zero share), coverage by year and the cross-tables follow Arshmehar's notebook
(04_bias_slicing.ipynb).

Each group is compared with a reference where one exists:
- time, calendar, weather, era and AFL slices: every hour of the study period, so a ratio
  below 1 means sensor outages or the sensor network's growth left that kind of hour short;
- precinct: the length of OSM walkable network in each CLUE area, so a ratio below 1 means
  fewer sensors than the precinct's share of the network;
- slope band: the length of surveyed footpath in each band.

ratio = the group's share of rows (or of sensors, for place slices) / its reference share, with
both shares taken over groups that have a reference. A group is flagged thin when it has fewer
than thin_min_sensors sensors or its ratio is below thin_ratio_below.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from src.bias.slices import (
    SENSOR_VALUES,
    UNKNOWN,
    afl_days,
    bins,
    load_bias_config,
    row_slices,
    sensor_precincts,
    sensor_slices,
)
from src.features.common import config_hash, load_features_config, log, write_meta
from src.features.counts_hourly import (
    read_weather,
    school_term_ranges,
    time_features,
    uni_ranges,
    weather_for_counts,
)
from src.features.sensor_match import sensor_locations_path
from src.features.sensor_steepness import outdoor_sensors, usable_sections
from src.ingest.common import REPO_ROOT, load_config, setup_logging

ROW_DIMS = [
    "time_of_day",
    "day_type",
    "public_holiday",
    "school_term",
    "uni_teaching",
    "rain",
    "temperature",
    "era",
    "afl_day",
    "source",
]
SENSOR_DIMS = [
    "precinct",
    "holdout_precinct",
    "volume_tier",
    "amenity_tier",
    "slope_band",
    "slope_source",
    "missing_hours",
    "match_review",
    "has_location",
    "sensor_start",
]
HOURLY_COLUMNS = [
    "location_id",
    "ts_utc",
    "date_local",
    "hour_local",
    "source",
    "pedestriancount",
    "suspect_outage_zero",
    "is_weekend",
    "is_public_holiday",
    "school_term",
    "uni_teaching",
    "wx_precipitation",
    "wx_temperature_2m",
]


def summarize(
    rows: pd.DataFrame,
    dim: str,
    reference: pd.Series | None,
    by_sensor: bool,
    cfg: dict[str, Any],
    kept: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-group counts and shares for one dimension.

    `rows` holds every hourly row with its slice columns and an `outage` flag. Shares and
    counts use the rows that are kept (not outage), except outage_share. Pass `kept`
    (rows without outages) when summarizing many dimensions, to filter only once.
    """
    if kept is None:
        kept = rows[~rows["outage"]]
    g = kept.groupby(dim, observed=True)
    out = pd.DataFrame(
        {
            "n_sensors": g["location_id"].nunique(),
            "n_rows": g.size(),
            "pedestrians": g["pedestriancount"].sum(),
            "mean_count": g["pedestriancount"].mean(),
            "median_count": g["pedestriancount"].median(),
            "p90_count": g["pedestriancount"].quantile(0.9),
        }
    )
    out["zero_share"] = (
        kept["pedestriancount"].eq(0).groupby(kept[dim], observed=True).mean()
    )
    out["outage_share"] = rows.groupby(dim, observed=True)["outage"].mean()
    out["row_share"] = out["n_rows"] / out["n_rows"].sum()
    out["sensor_share"] = out["n_sensors"] / kept["location_id"].nunique()
    out["pedestrian_share"] = out["pedestrians"] / out["pedestrians"].sum()

    out["reference_share"] = np.nan
    out["ratio"] = np.nan
    if reference is not None:
        ref = reference[reference.index != UNKNOWN]
        ref = ref / ref.sum()
        in_ref = out.index.isin(ref.index)
        basis = out["n_sensors"] if by_sensor else out["n_rows"]
        share = basis[in_ref] / basis[in_ref].sum()
        out["reference_share"] = ref.reindex(out.index)
        out.loc[in_ref, "ratio"] = share / out.loc[in_ref, "reference_share"]
    out["thin"] = (out["n_sensors"] < cfg["thin_min_sensors"]) | (
        out["ratio"] < cfg["thin_ratio_below"]
    )
    out = out.reset_index(names="group")
    out["group"] = out["group"].astype(str)
    out.insert(0, "dimension", dim)
    return out


def crosstab(kept: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Rows, share of all rows and median count for every combination of two slices."""
    g = kept.groupby([a, b], observed=True)["pedestriancount"]
    out = pd.DataFrame({"n_rows": g.size(), "median_count": g.median()})
    out["row_share"] = out["n_rows"] / len(kept)
    out = out.reset_index(names=["group_a", "group_b"])
    out[["group_a", "group_b"]] = out[["group_a", "group_b"]].astype(str)
    out.insert(0, "dimension_b", b)
    out.insert(0, "dimension_a", a)
    return out


def coverage_by_year(kept: pd.DataFrame, years: pd.Series) -> pd.DataFrame:
    """Rows, sensors and pedestrians per year: early years rest on few sensors."""
    g = kept.groupby(years.loc[kept.index].rename("year"))
    out = pd.DataFrame(
        {
            "n_rows": g.size(),
            "n_sensors": g["location_id"].nunique(),
            "pedestrians": g["pedestriancount"].sum(),
        }
    )
    out["row_share"] = out["n_rows"] / out["n_rows"].sum()
    return out.reset_index()


def reference_hours(
    first_utc: pd.Timestamp,
    last_utc: pd.Timestamp,
    tz: str,
    center: tuple,
    calendars: tuple,
    weather: pd.DataFrame,
    hourly_cfg: dict[str, Any],
    cfg: dict[str, Any],
    afl: pd.Series | None,
) -> pd.DataFrame:
    """The row slices for every hour of the study period, one row per hour."""
    ts = pd.Series(pd.date_range(first_utc, last_utc, freq="h"))
    hours = pd.DataFrame({"ts_utc": ts, "ts_local": ts.dt.tz_convert(tz)})
    holidays, school, uni = calendars
    feats = time_features(hours, holidays, school, uni, center, hourly_cfg)
    wx = weather_for_counts(ts, weather, hourly_cfg["weather_interval_vars"])
    return row_slices(pd.concat([feats, wx], axis=1), cfg, afl)


def network_reference(
    edges: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, cfg: dict[str, Any]
) -> pd.Series:
    """Walkable network length (m) per precinct, inside the CLUE areas only."""
    pts = edges[["length_m"]].set_geometry(edges.geometry.representative_point())
    j = gpd.sjoin(
        pts,
        blocks[["area_name", "geometry"]].to_crs(edges.crs),
        how="inner",
        predicate="within",
    )
    precinct = j["area_name"].replace(cfg["precinct_groups"])
    return j.groupby(precinct)["length_m"].sum()


def slope_reference(sections: gpd.GeoDataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Surveyed footpath length (m) per slope band."""
    band = bins(sections["gradepc"], cfg["slope_pct"])
    return sections.groupby(band)["distance"].sum()


def build(
    data_cfg: dict[str, Any], fcfg: dict[str, Any], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    site = cfg["site"]
    s = data_cfg["sites"][site]
    crs = fcfg["sites"][site]["crs"]
    raw = REPO_ROOT / data_cfg["raw_dir"]
    feats = REPO_ROOT / fcfg["processed_dir"] / "features" / site

    t = time.perf_counter()
    hourly = pd.read_parquet(feats / "counts_hourly.parquet", columns=HOURLY_COLUMNS)
    log.info("hourly table: %d rows in %.0f s", len(hourly), time.perf_counter() - t)

    afl_path = raw / "events" / "afl_attendance.csv"
    afl = None
    if afl_path.exists():
        afl = afl_days(pd.read_csv(afl_path), cfg["afl_venues"])
    else:
        log.warning("%s is missing: the afl_day slice is left out", afl_path)

    blocks = gpd.read_file(
        raw
        / "melbourne_portal"
        / "blocks-for-census-of-land-use-and-employment-clue.geojson"
    )
    sensors = outdoor_sensors(sensor_locations_path(data_cfg), crs)
    per_sensor = sensor_slices(
        hourly,
        sensor_precincts(sensors, blocks, cfg),
        pd.read_parquet(feats / "sensor_steepness.parquet"),
        pd.read_parquet(feats / "sensor_osm_features.parquet"),
        cfg,
    )

    rows = row_slices(hourly, cfg, afl)
    rows["source"] = hourly["source"].astype("string")
    rows["location_id"] = hourly["location_id"]
    rows["pedestriancount"] = hourly["pedestriancount"]
    rows["outage"] = (
        hourly["suspect_outage_zero"].to_numpy()
        if cfg["exclude_outage_rows"]
        else False
    )
    rows = rows.merge(
        per_sensor.drop(columns=SENSOR_VALUES), on="location_id", how="left"
    )
    dims = [d for d in ROW_DIMS + SENSOR_DIMS if d in rows.columns]
    rows[dims] = rows[dims].astype("category")

    hcfg = fcfg["counts_hourly"]
    cal = REPO_ROOT / "configs" / "calendars"
    holidays = pd.to_datetime(
        pd.read_csv(raw / "calendars" / f"{site}_public_holidays.csv")["date"]
    )
    ref_rows = reference_hours(
        hourly["ts_utc"].min(),
        hourly["ts_utc"].max(),
        s["timezone"],
        tuple(s["center"]),
        (
            holidays,
            school_term_ranges(cal / "vic_school_terms.yaml"),
            uni_ranges(cal / "unimelb_teaching_periods.yaml"),
        ),
        read_weather(raw / "weather", site),
        hcfg,
        cfg,
        afl,
    )
    steep = gpd.read_file(raw / "melbourne_portal" / "footpath-steepness.geojson")
    sections, _ = usable_sections(steep, crs, fcfg["sensor_steepness"]["min_section_m"])
    edges = gpd.read_parquet(
        REPO_ROOT / fcfg["interim_dir"] / "osm" / site / "graph" / "edges.parquet",
        columns=["length_m", "geometry"],
    )
    references = {dim: ref_rows[dim].value_counts() for dim in ref_rows.columns}
    references["precinct"] = network_reference(edges, blocks, cfg)
    references["slope_band"] = slope_reference(sections, cfg)

    kept = rows[~rows["outage"]]
    table = pd.concat(
        [
            summarize(rows, d, references.get(d), d in SENSOR_DIMS, cfg, kept)
            for d in dims
        ],
        ignore_index=True,
    )

    crosstabs = pd.concat(
        [crosstab(kept, a, b) for a, b in cfg["crosstabs"]], ignore_index=True
    )
    by_year = coverage_by_year(kept, pd.to_datetime(hourly["date_local"]).dt.year)

    thin = table[table["thin"]]
    meta = {
        "rows": len(rows),
        "rows_kept": int((~rows["outage"]).sum()),
        "sensors": int(rows["location_id"].nunique()),
        "reference_hours": len(ref_rows),
        "first_hour_utc": str(hourly["ts_utc"].min()),
        "last_hour_utc": str(hourly["ts_utc"].max()),
        "thin_groups": [f"{d}={g}" for d, g in zip(thin["dimension"], thin["group"])],
    }
    outputs = {
        "slice_representation": table,
        "slice_crosstabs": crosstabs,
        "coverage_by_year": by_year,
        "sensor_slices": per_sensor,
    }
    return outputs, meta


def main() -> None:
    setup_logging()
    data_cfg = load_config()
    fcfg = load_features_config()
    cfg = load_bias_config()
    t = time.perf_counter()
    outputs, meta = build(data_cfg, fcfg, cfg)
    out = REPO_ROOT / fcfg["processed_dir"] / "bias" / cfg["site"]
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.4f")
    path: Path = out / "slice_representation.csv"
    table = outputs["slice_representation"]
    write_meta(
        out / "slice_representation_meta.json",
        config_hash=config_hash(cfg),
        runtime_s=round(time.perf_counter() - t, 1),
        **meta,
    )
    log.info("wrote %s (%d groups); thin: %s", path, len(table), meta["thin_groups"])


if __name__ == "__main__":
    main()
