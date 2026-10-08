"""Feasibility check for the new-store before/after study: large openings near count sensors.

Usage: python -m src.analysis.new_openings [--sensor-radius METRES]

CLUE is an annual census, so an opening first seen in census year Y happened somewhere between
the Y-1 and Y surveys. The before window is therefore calendar year Y-2 and the after window is
Y+1; both need sensor counts. Writes data/processed/analysis/new_openings_candidates.csv.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml

from src.ingest.common import REPO_ROOT, load_config

PORTAL = "melbourne_portal"


def load_analysis_config() -> dict[str, Any]:
    with open(REPO_ROOT / "configs" / "analysis.yaml") as f:
        return yaml.safe_load(f)


def property_jumps(
    df: pd.DataFrame, value: str | None, threshold: float
) -> pd.DataFrame:
    """Rows (base_property_id, year, gain) where a property's total rose by >= threshold."""
    agg = (
        df.groupby(["base_property_id", "year"]).size()
        if value is None
        else df.groupby(["base_property_id", "year"])[value].sum()
    )
    wide = agg.unstack(fill_value=0).sort_index(axis=1)
    # Only compare consecutive census years that both exist.
    gain = wide.diff(axis=1)
    out = gain.stack().rename("gain").reset_index()
    return out[out.gain >= threshold]


def sensor_year_completeness(raw: Path) -> pd.Series:
    """Share of hours present per (sensor, calendar year), archive and live window combined."""
    with zipfile.ZipFile(raw / "counts_per_hour_2009-05_to_2022-12-14.csv.zip") as z:
        archive = pd.read_csv(z.open(z.namelist()[0]), usecols=["Sensor_ID", "Year"])
    archive = archive.rename(columns={"Sensor_ID": "sensor", "Year": "year"})
    live = pd.read_parquet(
        raw / "pedestrian-counting-system-monthly-counts-per-hour.parquet",
        columns=["location_id", "sensing_date"],
    )
    live = pd.DataFrame(
        {
            "sensor": live.location_id,
            "year": pd.to_datetime(live.sensing_date).dt.year,
        }
    )
    hours = pd.concat([archive, live]).groupby(["sensor", "year"]).size()
    return (hours / 8760).clip(upper=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-radius", type=float, help="override sensor_radius_m")
    args = ap.parse_args()

    cfg = load_config()
    acfg = load_analysis_config()
    p = acfg["new_openings"]
    if args.sensor_radius:
        p["sensor_radius_m"] = args.sensor_radius
    raw = REPO_ROOT / cfg["raw_dir"] / PORTAL

    biz = pd.read_parquet(
        raw / "business-establishments-with-address-and-industry-classification.parquet"
    )
    biz["year"] = pd.to_datetime(biz.census_year).dt.year
    biz = biz[biz.industry_anzsic4_code.str[:2].isin(p["attractor_divisions"])]
    seats = pd.read_parquet(raw / "cafes-and-restaurants-with-seating-capacity.parquet")
    seats["year"] = pd.to_datetime(seats.census_year).dt.year

    est = property_jumps(biz, None, p["min_new_establishments"])
    seat = property_jumps(seats, "number_of_seats", p["min_new_seats"])
    key = ["base_property_id", "year"]
    gains = pd.merge(
        est.rename(columns={"gain": "new_establishments"}),
        seat.rename(columns={"gain": "new_seats"}),
        on=key,
        how="outer",
    ).fillna({"new_establishments": 0, "new_seats": 0})

    # Locate each candidate, and describe it from the businesses there in year Y.
    loc = (
        pd.concat(
            [
                biz[key + ["longitude", "latitude"]],
                seats[key + ["longitude", "latitude"]],
            ]
        )
        .dropna()
        .groupby(key)[["longitude", "latitude"]]
        .median()
        .reset_index()
    )
    names = (
        pd.concat([biz, seats])
        .groupby(key)
        .agg(
            address=("business_address", "first"),
            example_tenants=(
                "trading_name",
                lambda t: "; ".join(t.dropna().unique()[:4]),
            ),
        )
        .reset_index()
    )
    sites = gains.merge(loc, on=key).merge(names, on=key, how="left")
    sites["site_id"] = range(len(sites))
    sites = gpd.GeoDataFrame(
        sites,
        geometry=gpd.points_from_xy(sites.longitude, sites.latitude),
        crs=4326,
    ).to_crs(p["crs"])

    # Net change in attractor establishments around each site, Y-1 -> Y (re-coding check).
    points = gpd.GeoDataFrame(
        biz[["year"]].dropna(),
        geometry=gpd.points_from_xy(biz.longitude, biz.latitude),
        crs=4326,
    ).to_crs(p["crs"])
    rings = sites[["site_id", "year", "geometry"]].copy()
    rings["geometry"] = rings.buffer(p["neighbourhood_radius_m"])
    hits = gpd.sjoin(points, rings, predicate="within", lsuffix="pt", rsuffix="site")
    now = hits[hits.year_pt == hits.year_site].groupby("site_id").size()
    before = hits[hits.year_pt == hits.year_site - 1].groupby("site_id").size()
    sites["neighbourhood_gain"] = sites.site_id.map(now).fillna(0) - sites.site_id.map(
        before
    ).fillna(0)
    sites["recoding_artefact"] = (sites.new_establishments > 0) & (
        sites.neighbourhood_gain < p["min_neighbourhood_gain"]
    )

    # Sensors close enough to see the opening, with counts in the before and after years.
    sensors = gpd.read_file(raw / "pedestrian-counting-system-sensor-locations.geojson")
    sensors = sensors[sensors.location_type == "Outdoor"].to_crs(p["crs"])
    complete = sensor_year_completeness(raw)
    sites["before_year"] = sites.year - 2
    sites["after_year"] = sites.year + 1
    # Every sensor within range of each site (a cross join is small: ~130 sites x 100 sensors).
    pairs = sites[["site_id", "before_year", "after_year", "geometry"]].merge(
        sensors[["location_id", "geometry"]], how="cross", suffixes=("", "_sensor")
    )
    pairs["dist"] = gpd.GeoSeries(pairs.geometry, crs=p["crs"]).distance(
        gpd.GeoSeries(pairs.geometry_sensor, crs=p["crs"])
    )
    pairs = pairs[pairs.dist <= p["sensor_radius_m"]].sort_values(["site_id", "dist"])
    lid = pairs.location_id.astype(int)
    pairs["usable"] = [
        complete.get((s, b), 0) >= p["min_year_completeness"]
        and complete.get((s, a), 0) >= p["min_year_completeness"]
        for s, b, a in zip(lid, pairs.before_year, pairs.after_year)
    ]
    pairs["label"] = (
        lid.astype(str) + " (" + pairs.dist.round().astype(int).astype(str) + " m)"
    )
    sites["sensors_within_radius"] = (
        sites.site_id.map(pairs.groupby("site_id").size()).fillna(0).astype(int)
    )
    sites["usable_sensors"] = sites.site_id.map(
        pairs[pairs.usable].groupby("site_id").label.agg(", ".join)
    ).fillna("")
    sites["covid_window"] = [
        bool({b, y, a} & set(p["covid_years"]))
        for b, y, a in zip(sites.before_year, sites.year, sites.after_year)
    ]
    sites["feasible"] = (sites.usable_sensors != "") & ~sites.recoding_artefact

    out_dir = REPO_ROOT / acfg["processed_dir"] / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "year",
        "address",
        "example_tenants",
        "new_establishments",
        "new_seats",
        "neighbourhood_gain",
        "recoding_artefact",
        "sensors_within_radius",
        "usable_sensors",
        "before_year",
        "after_year",
        "covid_window",
        "feasible",
    ]
    result = sites[cols].sort_values(["feasible", "year"], ascending=[False, True])
    suffix = "" if not args.sensor_radius else f"_{int(args.sensor_radius)}m"
    result.to_csv(out_dir / f"new_openings_candidates{suffix}.csv", index=False)

    print(
        f"candidates: {len(sites)} | re-coding artefacts: {int(sites.recoding_artefact.sum())} | "
        f"with a sensor within {p['sensor_radius_m']} m: {int((sites.sensors_within_radius > 0).sum())} | "
        f"feasible: {int(sites.feasible.sum())} "
        f"(outside COVID windows: {int((sites.feasible & ~sites.covid_window).sum())})"
    )
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(result[result.feasible].to_string(index=False))


if __name__ == "__main__":
    main()
