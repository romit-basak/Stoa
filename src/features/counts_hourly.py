"""Hourly training table for Melbourne: counts + time + weather + calendars + sensor match +
footpath slope.

Usage: python -m src.features.counts_hourly

One row per (sensor, hour). Writes data/processed/features/melbourne/counts_hourly.parquet and
counts_hourly_meta.json. The per-segment OSM features are not copied onto every row (that would
be ~115 columns x ~7M rows); join them on location_id from sensor_osm_features.parquet.

Counts come from the 2009-2022 archive, the live table, and a Wayback Machine copy of the portal
export for the gap between them (November 2022 to October 2024); each hour has one source.

Time. All the count tables use Melbourne's local clock, daylight saving included, and behaviour
follows that clock, so hour, weekday and calendar features are local. The Open-Meteo files use a
fixed UTC+10 offset all year, so weather is joined on UTC: joining on the local clock would put it
an hour out for half of every year. Two edges of the local clock are handled explicitly:
- the hour skipped when clocks go forward (02:00 on the first Sunday of October) cannot exist, so
  any count stamped with it is dropped;
- the hour repeated when clocks go back (02:00 on the first Sunday of April) appears once in the
  sources, so it is read as the first (daylight-saving) occurrence.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.features.common import (
    config_hash,
    load_features_config,
    log,
    write_meta,
    write_parquet,
)
from src.ingest.common import REPO_ROOT, load_config, setup_logging

ARCHIVE = "counts_per_hour_2009-05_to_2022-12-14.csv.zip"
LIVE = "pedestrian-counting-system-monthly-counts-per-hour.parquet"
WAYBACK = "counts_per_hour_wayback_2025-05-02.csv"


# --------------------------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------------------------


def read_archive(path: Path) -> tuple[pd.DataFrame, int]:
    """The 2009-2022 archive, with September 2010's hours recovered; returns (rows, rows re-houred)."""
    with zipfile.ZipFile(path) as z:
        a = pd.read_csv(
            z.open(z.namelist()[0]),
            usecols=["ID", "Sensor_ID", "Date_Time", "Hourly_Counts"],
            dtype={"ID": "int64", "Sensor_ID": "int32", "Hourly_Counts": "int32"},
        )
    out = pd.DataFrame(
        {
            "location_id": a["Sensor_ID"],
            "ts_local_naive": pd.to_datetime(
                a["Date_Time"], format="%B %d, %Y %I:%M:%S %p"
            ),
            "pedestriancount": a["Hourly_Counts"],
            "direction_1": np.nan,
            "direction_2": np.nan,
            "source": "archive",
        }
    )
    return rehour_midnight_days(out, a["ID"])


def rehour_midnight_days(
    counts: pd.DataFrame, row_id: pd.Series
) -> tuple[pd.DataFrame, int]:
    """Recover the hours of sensor-days whose 24 counts are all stamped midnight.

    In the archive every hour of September 2010 is stamped 00:00, so each sensor has 24 counts
    at midnight on each of those days (12,240 rows). The rows are still in hour order by the
    archive's ID, and in that order each day's counts trace the usual daily curve, so numbering
    them 0-23 by ID gives back the hour. Only groups of exactly 24 rows at midnight are
    renumbered; any other duplicate is left for drop_unplaceable. Port of the fix in Arshmehar's
    notebook (notebooks/02_combine.ipynb).
    """
    key = ["location_id", "ts_local_naive"]
    dup = counts.duplicated(key, keep=False)
    at_midnight = counts["ts_local_naive"].eq(counts["ts_local_naive"].dt.normalize())
    size = counts[dup].groupby(key)["pedestriancount"].transform("size")
    whole_day = dup & at_midnight & size.reindex(counts.index).eq(24)
    if not whole_day.any():
        return counts, 0
    days = counts[whole_day].assign(_id=row_id[whole_day]).sort_values("_id")
    hour = days.groupby(key).cumcount()
    out = counts.copy()
    out.loc[days.index, "ts_local_naive"] = days["ts_local_naive"] + pd.to_timedelta(
        hour, unit="h"
    )
    return out, int(whole_day.sum())


def read_live(path: Path) -> pd.DataFrame:
    live = pd.read_parquet(
        path,
        columns=[
            "location_id",
            "sensing_date",
            "hourday",
            "direction_1",
            "direction_2",
            "pedestriancount",
        ],
    )
    return pd.DataFrame(
        {
            "location_id": live["location_id"].astype("int32"),
            "ts_local_naive": pd.to_datetime(live["sensing_date"])
            + pd.to_timedelta(live["hourday"], unit="h"),
            "pedestriancount": live["pedestriancount"].astype("int32"),
            "direction_1": live["direction_1"].astype("float32"),
            "direction_2": live["direction_2"].astype("float32"),
            "source": "live",
        }
    )


def read_wayback(path: Path) -> pd.DataFrame:
    """The Wayback Machine copy of the portal's full CSV export (same columns as the live table)."""
    w = pd.read_csv(
        path,
        sep=";",
        usecols=[
            "location_id",
            "sensing_date",
            "hourday",
            "direction_1",
            "direction_2",
            "pedestriancount",
        ],
    )
    return pd.DataFrame(
        {
            "location_id": w["location_id"].astype("int32"),
            "ts_local_naive": pd.to_datetime(w["sensing_date"])
            + pd.to_timedelta(w["hourday"], unit="h"),
            "pedestriancount": w["pedestriancount"].astype("int32"),
            "direction_1": w["direction_1"].astype("float32"),
            "direction_2": w["direction_2"].astype("float32"),
            "source": "wayback",
        }
    )


def fill_gap(
    wayback: pd.DataFrame, archive: pd.DataFrame, live: pd.DataFrame
) -> pd.DataFrame:
    """Wayback rows strictly between the archive's last hour and the live table's first hour.

    The Wayback copy overlaps both tables; taking only the gap gives every hour a single source
    and leaves the archive and live rows as published.
    """
    after = archive["ts_local_naive"].max()
    before = live["ts_local_naive"].min()
    t = wayback["ts_local_naive"]
    return wayback[(t > after) & (t < before)]


def drop_unplaceable(counts: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop every row of a (sensor, local hour) that appears more than once.

    Such rows can't be put in order. September 2010's midnight stamps are fixed before this
    (rehour_midnight_days), so this is a safeguard: no other duplicates are known.
    """
    dup = counts.duplicated(["location_id", "ts_local_naive"], keep=False)
    return counts[~dup], int(dup.sum())


def localize(counts: pd.DataFrame, tz: str) -> tuple[pd.DataFrame, int]:
    """Attach local (tz-aware) and UTC timestamps; drop the non-existent spring-forward hour."""
    local = counts["ts_local_naive"].dt.tz_localize(
        tz,
        # The repeated autumn hour appears once in the data: take the first (daylight) occurrence.
        ambiguous=np.ones(len(counts), dtype=bool),
        nonexistent="NaT",
    )
    keep = local.notna()
    out = counts[keep].copy()
    out["ts_local"] = local[keep]
    out["ts_utc"] = out["ts_local"].dt.tz_convert("UTC")
    return out.drop(columns="ts_local_naive"), int((~keep).sum())


def flag_outages(
    counts: pd.DataFrame, min_run: int, day_hours: tuple[int, int]
) -> pd.Series:
    """True for hours inside a run of >= min_run consecutive zero counts that touches daytime.

    Runs are within one sensor and must be consecutive in time, so a gap in the record ends a run.
    """
    c = counts.sort_values(["location_id", "ts_utc"])
    zero = c["pedestriancount"].eq(0)
    step = c.groupby("location_id")["ts_utc"].diff().ne(pd.Timedelta(hours=1))
    new_sensor = c["location_id"].ne(c["location_id"].shift())
    # A run id changes whenever the zero/non-zero state flips, the sensor changes or time jumps.
    run = (zero.ne(zero.shift()) | step | new_sensor).cumsum()
    hour = c["ts_local"].dt.hour
    daytime = hour.between(day_hours[0], day_hours[1] - 1)
    g = pd.DataFrame({"run": run, "zero": zero, "day": daytime})
    length = g.groupby("run")["zero"].transform("size")
    touches_day = g.groupby("run")["day"].transform("any")
    flagged = zero & (length >= min_run) & touches_day
    return flagged.reindex(counts.index)


# --------------------------------------------------------------------------------------------
# Time, calendars and sun
# --------------------------------------------------------------------------------------------


def solar_elevation(ts_utc: pd.Series, lat: float, lon: float) -> np.ndarray:
    """Sun elevation in degrees (NOAA's simplified solar position; accurate to well under 1 degree)."""
    t = pd.DatetimeIndex(ts_utc)
    doy = t.dayofyear.to_numpy()
    hours = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    gamma = 2 * np.pi / 365 * (doy - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    true_solar_min = hours * 60 + eqtime + 4 * lon
    hour_angle = np.radians(true_solar_min / 4 - 180)
    phi = np.radians(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(
        hour_angle
    )
    return 90 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))


def in_ranges(dates: pd.Series, ranges: list[tuple[str, str]]) -> pd.Series:
    """True where a date falls in any inclusive [start, end] range."""
    out = pd.Series(False, index=dates.index)
    for start, end in ranges:
        out |= dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return out


def covered(dates: pd.Series, flag: pd.Series, years: set[int]) -> pd.Series:
    """The flag where the calendar covers the date's year, missing (unknown) elsewhere."""
    out = flag.astype("boolean")
    out[~dates.dt.year.isin(years)] = pd.NA
    return out


def school_term_ranges(path: Path) -> tuple[list[tuple[str, str]], set[int]]:
    terms = yaml.safe_load(path.read_text())["terms"]
    ranges = [
        (str(t.get("students_start", t["start"])), str(t["end"]))
        for year in terms.values()
        for t in year
    ]
    return ranges, {int(y) for y in terms}


def uni_ranges(path: Path) -> tuple[dict[str, list[tuple[str, str]]], set[int]]:
    sems = yaml.safe_load(path.read_text())["semesters"]
    out: dict[str, list[tuple[str, str]]] = {"teaching": [], "break": [], "exams": []}
    for year in sems.values():
        for sem in year.values():
            for key, ranges in out.items():
                if key in sem:
                    start, end = sem[key]
                    ranges.append((str(start), str(end)))
    return out, {int(y) for y in sems}


def time_features(
    counts: pd.DataFrame,
    holidays: pd.Series,
    school: tuple[list, set[int]],
    uni: tuple[dict, set[int]],
    center: tuple,
    cfg: dict,
) -> pd.DataFrame:
    """Local-clock calendar features, sun elevation and period flags.

    School and university flags are missing (not False) in years their calendar doesn't cover.
    """
    local = counts["ts_local"]
    date = local.dt.tz_localize(None).dt.normalize()
    out = pd.DataFrame(index=counts.index)
    out["date_local"] = date
    out["hour_local"] = local.dt.hour.astype("int8")
    out["weekday"] = local.dt.dayofweek.astype("int8")  # Monday = 0
    out["hour_of_week"] = (
        out["weekday"].astype("int16") * 24 + out["hour_local"]
    ).astype("int16")
    out["is_weekend"] = out["weekday"] >= 5
    out["is_public_holiday"] = date.isin(holidays)
    terms, term_years = school
    periods, uni_years = uni
    out["school_term"] = covered(date, in_ranges(date, terms), term_years)
    teaching = in_ranges(date, periods["teaching"]) & ~in_ranges(date, periods["break"])
    out["uni_teaching"] = covered(date, teaching, uni_years)
    out["uni_exams"] = covered(date, in_ranges(date, periods["exams"]), uni_years)
    # Sun at the middle of the count's hour; physics uses UTC, not the local clock.
    mid = counts["ts_utc"] + pd.Timedelta(minutes=30)
    out["sun_elevation_deg"] = solar_elevation(mid, center[1], center[0]).astype(
        "float32"
    )
    covid = cfg["covid_period"]
    out["covid_period"] = date.between(pd.Timestamp(covid[0]), pd.Timestamp(covid[1]))
    out["post_metro_tunnel"] = date >= pd.Timestamp(cfg["metro_tunnel_opened"])
    return out


# --------------------------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------------------------


def read_weather(folder: Path, site: str) -> pd.DataFrame:
    """All yearly Open-Meteo files as one table indexed by UTC hour."""
    frames = []
    for path in sorted(folder.glob(f"{site}_openmeteo_*.json")):
        w = json.loads(path.read_text())
        df = pd.DataFrame(w["hourly"])
        # Open-Meteo's local times carry one fixed offset for the whole file.
        df["ts_utc"] = (
            pd.to_datetime(df.pop("time"))
            - pd.Timedelta(seconds=w["utc_offset_seconds"])
        ).dt.tz_localize("UTC")
        frames.append(df)
    weather = (
        pd.concat(frames).drop_duplicates("ts_utc", keep="last").set_index("ts_utc")
    )
    return weather.sort_index()


def weather_for_counts(
    ts_utc: pd.Series, weather: pd.DataFrame, interval_vars: list[str]
) -> pd.DataFrame:
    """Weather for each count hour [h, h+1): instants at h, preceding-hour values from h+1."""
    instant = [c for c in weather.columns if c not in interval_vars]
    at_start = weather[instant].reindex(ts_utc.to_numpy())
    at_end = weather[[c for c in interval_vars if c in weather.columns]].reindex(
        (ts_utc + pd.Timedelta(hours=1)).to_numpy()
    )
    out = pd.concat(
        [at_start.reset_index(drop=True), at_end.reset_index(drop=True)], axis=1
    )
    out.columns = [f"wx_{c}" for c in out.columns]
    out.index = ts_utc.index
    return out.astype("float32")


# --------------------------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------------------------


def build(
    data_cfg: dict[str, Any], fcfg: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = fcfg["counts_hourly"]
    site = cfg["site"]
    s = data_cfg["sites"][site]
    raw = REPO_ROOT / data_cfg["raw_dir"]
    portal = raw / "melbourne_portal"

    t = time.perf_counter()
    (archive, n_rehoured), live = (
        read_archive(portal / ARCHIVE),
        read_live(portal / LIVE),
    )
    parts = [archive, live]
    if (portal / WAYBACK).exists():
        parts.append(fill_gap(read_wayback(portal / WAYBACK), archive, live))
    else:
        log.warning(
            "%s not held, so the gap between the archive and the live table has no counts "
            "(python -m src.ingest.melbourne_portal --only "
            "pedestrian-counting-system-monthly-counts-per-hour fetches it)",
            WAYBACK,
        )
    counts = pd.concat(parts, ignore_index=True)
    n_read = len(counts)
    counts, n_dup = drop_unplaceable(counts)
    counts, n_nonexistent = localize(counts, s["timezone"])
    counts = counts.reset_index(drop=True)
    log.info("counts: %d rows read in %.0f s", n_read, time.perf_counter() - t)

    counts["suspect_outage_zero"] = flag_outages(
        counts, cfg["outage_zero_run_hours"], tuple(cfg["outage_daytime_hours"])
    )

    cal = REPO_ROOT / "configs" / "calendars"
    holidays = pd.to_datetime(
        pd.read_csv(raw / "calendars" / f"{site}_public_holidays.csv")["date"]
    )
    feats = time_features(
        counts,
        holidays,
        school_term_ranges(cal / "vic_school_terms.yaml"),
        uni_ranges(cal / "unimelb_teaching_periods.yaml"),
        tuple(s["center"]),
        cfg,
    )
    weather = read_weather(raw / "weather", site)
    wx = weather_for_counts(counts["ts_utc"], weather, cfg["weather_interval_vars"])

    sensors_path = (
        REPO_ROOT
        / fcfg["processed_dir"]
        / "features"
        / site
        / "sensor_osm_features.parquet"
    )
    sensors = pd.read_parquet(
        sensors_path,
        columns=[
            "location_id",
            "sensor_description",
            "location_type",
            "segment_id",
            "match_method",
            "needs_review",
        ],
    )
    steepness = pd.read_parquet(
        sensors_path.with_name("sensor_steepness.parquet"),
        columns=[
            "location_id",
            "slope_pct",
            "slope_pct_local_mean",
            "elevation_m",
            "place_source",
        ],
    )
    table = (
        pd.concat([counts, feats, wx], axis=1)
        .merge(sensors, on="location_id", how="left")
        .merge(steepness, on="location_id", how="left")
    )
    table[["slope_pct", "slope_pct_local_mean", "elevation_m"]] = table[
        ["slope_pct", "slope_pct_local_mean", "elevation_m"]
    ].astype("float32")
    table["has_location"] = table["location_type"].notna()
    table["needs_review"] = table["needs_review"].astype("boolean")
    table[["direction_1", "direction_2"]] = table[
        ["direction_1", "direction_2"]
    ].astype("float32")
    table["source"] = table["source"].astype("category")
    table = table.sort_values(["ts_utc", "location_id"]).reset_index(drop=True)

    first = [
        "location_id",
        "ts_utc",
        "ts_local",
        "date_local",
        "hour_local",
        "source",
        "pedestriancount",
    ]
    table = table[first + [c for c in table.columns if c not in first]]
    no_wx = table["wx_temperature_2m"].isna()
    meta = {
        "rows_read": n_read,
        "rows": len(table),
        "rows_by_source": {
            str(k): int(v) for k, v in table["source"].value_counts().items()
        },
        "archive_rows_rehoured": n_rehoured,
        "dropped_unplaceable_duplicates": n_dup,
        "dropped_nonexistent_spring_forward": n_nonexistent,
        "sensors": int(table["location_id"].nunique()),
        "sensors_without_location": sorted(
            table.loc[~table["has_location"], "location_id"].unique().tolist()
        ),
        "rows_without_location": int((~table["has_location"]).sum()),
        "rows_suspect_outage_zero": int(table["suspect_outage_zero"].sum()),
        "rows_without_weather": int(no_wx.sum()),
        "first_hour_local": str(table["ts_local"].min()),
        "last_hour_local": str(table["ts_local"].max()),
    }
    return table, meta


def main() -> None:
    setup_logging()
    data_cfg = load_config()
    fcfg = load_features_config()
    t = time.perf_counter()
    table, meta = build(data_cfg, fcfg)
    out = REPO_ROOT / fcfg["processed_dir"] / "features" / fcfg["counts_hourly"]["site"]
    write_parquet(table, out / "counts_hourly.parquet")
    write_meta(
        out / "counts_hourly_meta.json",
        config_hash=config_hash(fcfg["counts_hourly"]),
        runtime_s=round(time.perf_counter() - t, 1),
        **meta,
    )
    log.info("counts_hourly: %s", meta)


if __name__ == "__main__":
    main()
