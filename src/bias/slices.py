"""Slice definitions for bias analysis: which group each hourly row and each sensor falls in.

Settings are in configs/bias.yaml. The same slices feed the representation report
(src/bias/representation.py) and, later, the per-slice model errors.

The data has no demographic attributes: Stoa holds no personal data (CLAUDE.md, core rule 6).
Bias here means the data, or a model, serving some places, times and conditions worse than
others, so the slices are places, times, weather, eras, data sources and sensor quality.

The slice list follows Arshmehar's proposal and her notebook (04_bias_slicing.ipynb): precinct,
sensor volume, time of day, weekday or weekend, weather, era, data source, slope and data
quality, plus the held-out sensor lists and the sensor start period. Calendar flags, AFL
days, the amenity tier, the match-review flag and the place references were added here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from src.ingest.common import REPO_ROOT

UNKNOWN = "unknown"


def load_bias_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or REPO_ROOT / "configs" / "bias.yaml") as f:
        return yaml.safe_load(f)


def bins(values: pd.Series, spec: dict[str, list]) -> pd.Series:
    """Label values by [edge_i, edge_i+1) bins; the last bin is open. Missing -> unknown."""
    edges = [*spec["edges"], np.inf]
    out = pd.cut(values, edges, right=False, labels=spec["labels"])
    return out.astype("string").fillna(UNKNOWN)


def time_of_day(hour: pd.Series, spec: dict[str, list[int]]) -> pd.Series:
    out = pd.Series(UNKNOWN, index=hour.index, dtype="string")
    for name, (start, end) in spec.items():
        out[(hour >= start) & (hour < end)] = name
    return out


def era(dates: pd.Series, eras: dict[str, str]) -> pd.Series:
    """Era of each date; an era runs from its start date to the next era's start."""
    starts = sorted((pd.Timestamp(d), name) for name, d in eras.items())
    out = pd.Series(UNKNOWN, index=dates.index, dtype="string")
    for ts, name in starts:
        out[dates >= ts] = name
    return out


def flag_label(flag: pd.Series, yes: str, no: str) -> pd.Series:
    """True/False/NA -> yes/no/unknown."""
    f = flag.astype("boolean")
    out = pd.Series(UNKNOWN, index=flag.index, dtype="string")
    out[f.fillna(False)] = yes
    out[(~f).fillna(False)] = no
    return out


def afl_days(attendance: pd.DataFrame, venues: dict[str, str]) -> pd.Series:
    """Local date -> stadium label for AFL match days at the given venues ('both' if both)."""
    a = attendance[attendance["venue"].isin(venues)]
    by_day = a.groupby(pd.to_datetime(a["date"]))["venue"].agg(
        lambda v: "both" if v.nunique() > 1 else venues[v.iloc[0]]
    )
    by_day.index.name = "date_local"
    return by_day


def row_slices(
    frame: pd.DataFrame, cfg: dict[str, Any], afl: pd.Series | None
) -> pd.DataFrame:
    """Time, calendar, weather and era slices for rows of the hourly table (or plain hours)."""
    out = pd.DataFrame(index=frame.index)
    out["time_of_day"] = time_of_day(frame["hour_local"], cfg["time_of_day"])
    out["day_type"] = flag_label(frame["is_weekend"], "weekend", "weekday")
    out["public_holiday"] = flag_label(
        frame["is_public_holiday"], "holiday", "not_holiday"
    )
    out["school_term"] = flag_label(frame["school_term"], "term", "school_holidays")
    out["uni_teaching"] = flag_label(frame["uni_teaching"], "teaching", "not_teaching")
    out["rain"] = bins(frame["wx_precipitation"], cfg["rain_mm"])
    out["temperature"] = bins(frame["wx_temperature_2m"], cfg["temperature_c"])
    out["era"] = era(frame["date_local"], cfg["eras"])
    if afl is not None:
        out["afl_day"] = frame["date_local"].map(afl).astype("string").fillna("none")
    return out


def sensor_precincts(
    sensors: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Precinct (grouped CLUE area) and held-out precinct of each located sensor."""
    j = gpd.sjoin(
        sensors[["location_id", "geometry"]],
        blocks[["area_name", "geometry"]].to_crs(sensors.crs),
        how="left",
        predicate="within",
    ).drop_duplicates("location_id")
    area = j["area_name"].astype("string")
    out = pd.DataFrame({"location_id": j["location_id"].to_numpy()})
    out["precinct"] = (
        area.replace(cfg["precinct_groups"]).fillna("outside_clue").to_numpy()
    )
    holdout = pd.Series("training_area", index=j.index, dtype="string")
    for name, ids in cfg["holdout_precincts"].items():
        holdout[j["location_id"].isin(ids)] = name
    out["holdout_precinct"] = holdout.to_numpy()
    return out


def missing_hours_share(hourly: pd.DataFrame) -> pd.Series:
    """Share of hours with no row, per sensor, within each source's span for that sensor.

    Spans are taken per source so the 2023-24 gap between the archive and the live table
    (missing for every sensor) doesn't count against each sensor.
    """
    g = hourly.groupby(["location_id", "source"], observed=True)["ts_utc"]
    span = (g.max() - g.min()) / pd.Timedelta(hours=1) + 1
    present = g.size()
    per_sensor = (
        pd.DataFrame({"span": span, "present": present})
        .groupby("location_id")[["span", "present"]]
        .sum()
    )
    return 1 - per_sensor["present"] / per_sensor["span"]


def tiers(values: pd.Series, labels: list[str]) -> pd.Series:
    """Equal-sized tiers by value (ties broken by order), labelled low to high."""
    ranks = values.rank(method="first")
    return pd.qcut(ranks, len(labels), labels=labels).astype("string")


SENSOR_VALUES = ["mean_count", "first_year", "missing_share", "slope_pct"]


def sensor_slices(
    hourly: pd.DataFrame,
    precincts: pd.DataFrame,
    steepness: pd.DataFrame,
    osm: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """One row per sensor in the hourly table, with each sensor-level slice."""
    clean = hourly[~hourly["suspect_outage_zero"]]
    by_sensor = clean.groupby("location_id")
    s = pd.DataFrame(
        {
            "mean_count": by_sensor["pedestriancount"].mean(),
            "first_year": by_sensor["date_local"].min().dt.year,
        }
    )
    s["missing_share"] = missing_hours_share(hourly)
    s = s.reset_index()
    s = s.merge(precincts, on="location_id", how="left")
    s = s.merge(
        steepness[["location_id", "slope_pct", "place_source"]],
        on="location_id",
        how="left",
    )
    r = cfg["amenity_radius_m"]
    poi = [c for c in osm.columns if c.startswith("poi_") and c.endswith(f"_{r}")]
    amen = osm[["location_id", "needs_review"]].assign(amenities=osm[poi].sum(axis=1))
    s = s.merge(amen, on="location_id", how="left")

    located = s["precinct"].notna()
    out = pd.DataFrame({"location_id": s["location_id"]})
    out["has_location"] = np.where(located, "located", "no_location")
    out["precinct"] = s["precinct"].astype("string").fillna("no_location")
    out["holdout_precinct"] = (
        s["holdout_precinct"].astype("string").fillna("no_location")
    )
    out["volume_tier"] = tiers(s["mean_count"], cfg["volume_tiers"])
    out["amenity_tier"] = pd.Series("no_location", index=s.index, dtype="string")
    out.loc[located, "amenity_tier"] = tiers(
        s.loc[located, "amenities"], cfg["amenity_tiers"]
    )
    out["slope_band"] = bins(s["slope_pct"], cfg["slope_pct"])
    out["slope_source"] = s["place_source"].astype("string").fillna(UNKNOWN)
    out["missing_hours"] = bins(s["missing_share"], cfg["missing_hours_share"])
    out["match_review"] = flag_label(s["needs_review"], "needs_review", "auto_ok")
    for col in ("slope_band", "slope_source", "match_review"):
        out.loc[~located, col] = "no_location"
    # When the sensor started counting: early years rest on a handful of streets.
    out["sensor_start"] = bins(s["first_year"], cfg["sensor_start_year"])
    # Values behind the tiers, kept for the per-sensor table (not slices themselves).
    for col in ("mean_count", "first_year", "missing_share", "slope_pct"):
        out[col] = s[col].to_numpy()
    return out
