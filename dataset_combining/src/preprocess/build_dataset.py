"""Unify, clean and join the raw sources into one row per sensor-hour.

Place features (slope, elevation) join by location_id; time features (ERA5
weather, city microclimate) join by timestamp_utc. Joins only add columns.

Outputs
  data/interim/counts_clean.parquet      unified counts, incl. direction columns
  data/interim/missing_hours.parquet     per-sensor gaps (absent rows, never zeros)
  data/processed/sensor_table.parquet    one row per sensor
  data/processed/hour_table.parquet      one row per UTC hour
  data/processed/pcs_hourly_joined.parquet   the training table
  data/processed/qa_report.json          every check and count below

Usage:  python -I src/preprocess/build_dataset.py [--sensors-only]
"""
from __future__ import annotations

import argparse
import calendar
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests
import shapely
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "data" / "processed"
TZ = "Australia/Melbourne"
KEY = ["location_id", "sensing_date", "hourday"]

# When sources overlap on a sensor-hour, the newest official copy wins.
SOURCE_PRIORITY = {"live": 0, "archive": 1, "wayback": 2}
WAYBACK_COUNTS = "pcs_counts_wayback_20250502183548.csv"

# Footpath match: ignore sections shorter than this (their grades are noise,
# e.g. 0.5 m slivers reporting 500 %), and report how far the match is.
MIN_SECTION_M = 5.0
LOCAL_SLOPE_RADIUS_M = 25.0
MAX_MATCH_M = 25.0

# Sensors with no footpath section within MAX_MATCH_M take elevation and slope
# from the GA 5 m LiDAR DEM instead (slope = terrain gradient over +/-10 m).
GA_DEM_IDENTIFY = "https://services.ga.gov.au/gis/rest/services/DEM_LiDAR_5m_2025/MapServer/identify"
DEM_STEP_M = 10.0
DEM_CACHE = RAW / "ga_dem_5m_samples.json"

# Open-Meteo stamps accumulated/"preceding hour" variables at the END of the
# hour. Counts are stamped at the START, so these are shifted back one hour.
WEATHER_INSTANT = {
    "temperature_2m": "temp_c",
    "apparent_temperature": "apparent_temp_c",
    "dew_point_2m": "dew_point_c",
    "relative_humidity_2m": "rel_humidity_pct",
    "cloud_cover": "cloud_cover_pct",
    "wind_speed_10m": "wind_speed_kmh",
    "wind_direction_10m": "wind_dir_deg",
    "surface_pressure": "surface_pressure_hpa",
    "is_day": "is_day",
}
WEATHER_PRECEDING = {
    "precipitation": "precip_mm",
    "rain": "rain_mm",
    "sunshine_duration": "sunshine_s",
    "shortwave_radiation": "shortwave_rad_wm2",
    "wind_gusts_10m": "wind_gust_kmh",
    "weather_code": "weather_code",
}

# City microclimate: street-level devices only, implausible readings dropped.
MC_VARS = {
    "airtemperature": ("mc_air_temp_c", -5, 50),
    "relativehumidity": ("mc_rel_humidity_pct", 0, 100),
    "atmosphericpressure": ("mc_pressure_hpa", 950, 1050),
    "averagewindspeed": ("mc_wind_speed_ms", 0, 40),
    "gustwindspeed": ("mc_wind_gust_ms", 0, 60),
    "pm25": ("mc_pm25_ugm3", 0, 1000),
    "pm10": ("mc_pm10_ugm3", 0, 1000),
    "noise": ("mc_noise_db", 30, 130),
}

qa: dict = {}


# ---------------------------------------------------------------- counts ----
def load_archive() -> pd.DataFrame:
    with zipfile.ZipFile(RAW / "pcs_archive_2009_2022.csv.zip") as z:
        name = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        with z.open(name) as f:
            cols = ["ID", "Year", "Month", "Mdate", "Time", "Sensor_ID", "Hourly_Counts"]
            a = pacsv.read_csv(f, convert_options=pacsv.ConvertOptions(include_columns=cols)).to_pandas()
    month = a["Month"].map({m: i for i, m in enumerate(calendar.month_name) if m}).astype("int64")
    a["sensing_date"] = pd.to_datetime(pd.DataFrame({"year": a.Year, "month": month, "day": a.Mdate}))

    # Sep 2010 is stamped 00:00 on every row (the 11,730 "repeated hours"). The
    # 24 rows per sensor-day are hours 0..23 in ID order, so recover them.
    dup = a.duplicated(["Sensor_ID", "sensing_date", "Time"], keep=False)
    bad = a.loc[dup].sort_values("ID")
    groups = bad.groupby(["Sensor_ID", "sensing_date"])
    assert bad["Time"].eq(0).all() and groups.size().eq(24).all(), "unexpected archive duplicates"
    a.loc[bad.index, "Time"] = groups.cumcount()
    qa["archive_rows"] = len(a)
    qa["archive_sep2010_rows_rehoured"] = int(len(bad))
    qa["archive_sep2010_extra_rows"] = int(len(bad) - groups.ngroups)

    return pd.DataFrame({
        "location_id": a.Sensor_ID.astype("int16"),
        "sensing_date": a.sensing_date,
        "hourday": a.Time.astype("int8"),
        "pedestriancount": a.Hourly_Counts.astype("int32"),
        "direction_1": pd.Series(pd.NA, index=a.index, dtype="Int32"),
        "direction_2": pd.Series(pd.NA, index=a.index, dtype="Int32"),
        "source": "archive",
    })


def load_live() -> pd.DataFrame:
    # Every dated pull is kept; for a repeated sensor-hour the newest pull wins.
    frames = []
    for p in sorted(RAW.glob("pcs_live_counts_*.parquet"), reverse=True):
        cols = ["location_id", "sensing_date", "hourday", "direction_1", "direction_2", "pedestriancount"]
        frames.append(pd.read_parquet(p, columns=cols))
    live = pd.concat(frames, ignore_index=True).drop_duplicates(KEY, keep="first")
    qa["live_rows"] = len(live)
    qa["live_pulls"] = len(frames)
    return pd.DataFrame({
        "location_id": live.location_id.astype("int16"),
        "sensing_date": pd.to_datetime(live.sensing_date),
        "hourday": live.hourday.astype("int8"),
        "pedestriancount": live.pedestriancount.astype("int32"),
        "direction_1": live.direction_1.astype("Int32"),
        "direction_2": live.direction_2.astype("Int32"),
        "source": "live",
    })


def load_wayback() -> pd.DataFrame:
    cols = ["location_id", "sensing_date", "hourday", "direction_1", "direction_2", "pedestriancount"]
    w = pacsv.read_csv(
        RAW / WAYBACK_COUNTS,
        parse_options=pacsv.ParseOptions(delimiter=";"),
        convert_options=pacsv.ConvertOptions(include_columns=cols),
    ).to_pandas()
    qa["wayback_rows"] = len(w)
    return pd.DataFrame({
        "location_id": w.location_id.astype("int16"),
        "sensing_date": pd.to_datetime(w.sensing_date),
        "hourday": w.hourday.astype("int8"),
        "pedestriancount": w.pedestriancount.astype("int32"),
        "direction_1": w.direction_1.astype("Int32"),
        "direction_2": w.direction_2.astype("Int32"),
        "source": "wayback",
    })


def unify_counts() -> pd.DataFrame:
    c = pd.concat([load_live(), load_archive(), load_wayback()], ignore_index=True)
    c["source"] = c["source"].astype("category")
    c["prio"] = c["source"].map(SOURCE_PRIORITY).astype("int8")
    c = c.sort_values(KEY + ["prio"]).drop_duplicates(KEY, keep="first").drop(columns="prio")
    qa["rows_after_source_merge"] = len(c)
    qa["rows_by_source"] = c["source"].value_counts().to_dict()

    # Wall-clock Melbourne time -> UTC. Spring-forward 02:00 does not exist
    # (archive rows there are mostly 0 fillers): drop. Fall-back 02:00 appears
    # once per sensor: treat it as the first (daylight-time) occurrence.
    local = pd.DatetimeIndex(c["sensing_date"] + pd.to_timedelta(c["hourday"], unit="h"))
    aware = local.tz_localize(TZ, ambiguous=np.ones(len(local), dtype=bool), nonexistent="NaT")
    keep = ~aware.isna()
    qa["dst_nonexistent_rows_dropped"] = int((~keep).sum())
    qa["dst_ambiguous_rows_assigned_daylight_time"] = int(
        (local.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").isna() & keep).sum())
    c = c.loc[keep].copy()
    c["timestamp_local"] = aware[keep].astype(f"datetime64[ns, {TZ}]")
    c["timestamp_utc"] = c["timestamp_local"].dt.tz_convert("UTC")
    assert not c.duplicated(["location_id", "timestamp_utc"]).any(), "duplicate sensor-hours in UTC"
    return c.sort_values(["location_id", "timestamp_utc"]).reset_index(drop=True)


def clean_counts(c: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    qa["negative_counts"] = int((c.pedestriancount < 0).sum())
    c = c[c.pedestriancount >= 0]
    has_dir = c.direction_1.notna() & c.direction_2.notna()
    qa["direction_rows_checked"] = int(has_dir.sum())
    qa["direction_sum_mismatches"] = int(
        (c.direction_1[has_dir] + c.direction_2[has_dir] != c.pedestriancount[has_dir]).sum())

    indoor = sensors.loc[sensors.location_type.eq("Indoor"), "location_id"]
    no_coords = sensors.loc[sensors.latitude.isna(), "location_id"]
    unknown = ~c.location_id.isin(sensors.location_id)
    drop = c.location_id.isin(indoor) | c.location_id.isin(no_coords) | unknown
    qa["rows_dropped_indoor"] = int(c.location_id.isin(indoor).sum())
    qa["rows_dropped_no_coordinates"] = int((c.location_id.isin(no_coords) | unknown).sum())
    c = c[~drop].copy()

    # Long all-zero runs at a street sensor are outages logged as 0. Flag them so
    # they can be treated as missing; nothing is deleted here.
    z = c.pedestriancount.eq(0)
    step = c.timestamp_utc.diff().ne(pd.Timedelta(hours=1))
    run = (z.ne(z.shift()) | c.location_id.ne(c.location_id.shift()) | step).cumsum()
    run_len = run.map(run.value_counts())
    c["suspect_outage_zero"] = z & (run_len >= 24)
    qa["rows_flagged_suspect_outage_zero"] = int(c.suspect_outage_zero.sum())
    qa["rows_clean"] = len(c)
    return c.reset_index(drop=True)


def missing_hours(c: pd.DataFrame) -> pd.DataFrame:
    prev = c.groupby("location_id").timestamp_utc.shift()
    gap = (c.timestamp_utc - prev) > pd.Timedelta(hours=1)
    g = pd.DataFrame({
        "location_id": c.location_id[gap],
        "gap_start_utc": prev[gap] + pd.Timedelta(hours=1),
        "gap_end_utc": c.timestamp_utc[gap] - pd.Timedelta(hours=1),
    })
    g["n_hours"] = ((g.gap_end_utc - g.gap_start_utc) / pd.Timedelta(hours=1) + 1).astype("int64")
    qa["missing_hour_gaps"] = len(g)
    qa["missing_hours_total"] = int(g.n_hours.sum())
    return g.reset_index(drop=True)


# --------------------------------------------------------------- sensors ----
def sensor_coordinates(location_ids) -> pd.DataFrame:
    cur = pd.read_parquet(RAW / "pcs_sensor_locations.parquet").drop(columns="location")
    cur["coord_source"] = "sensor_locations_current"

    # Retired sensors are gone from the current table; take them from the
    # Oct 2022 copy of the same City of Melbourne table.
    old = pd.DataFrame(json.loads((RAW / "pcs_sensor_locations_2022-10-24_wayback.json").read_text("utf-8")))
    old = pd.DataFrame({
        "location_id": old.sensor_id.astype(int),
        "sensor_description": old.sensor_description,
        "sensor_name": old.sensor_name,
        "installation_date": pd.to_datetime(old.installation_date).dt.date,
        "status": old.status,
        "latitude": old.latitude.astype(float),
        "longitude": old.longitude.astype(float),
        "coord_source": "sensor_locations_2022_snapshot",
    })
    s = pd.concat([cur, old[~old.location_id.isin(cur.location_id)]], ignore_index=True)

    # Last resort: the per-row coordinates carried in the newer count exports.
    missing = sorted(set(location_ids) - set(s.location_id))
    if missing:
        w = pacsv.read_csv(RAW / WAYBACK_COUNTS, parse_options=pacsv.ParseOptions(delimiter=";"),
                           convert_options=pacsv.ConvertOptions(include_columns=["location_id", "sensor_name", "location"])
                           ).to_pandas().dropna().drop_duplicates("location_id")
        w = w[w.location_id.isin(missing)]
    if missing and len(w):
        latlon = w.location.str.split(",", expand=True).astype(float)
        s = pd.concat([s, pd.DataFrame({
            "location_id": w.location_id, "sensor_name": w.sensor_name,
            "latitude": latlon[0], "longitude": latlon[1], "coord_source": "count_export_rows",
        })], ignore_index=True)
    s = s[s.location_id.isin(location_ids)].copy()
    s["location_id"] = s.location_id.astype("int16")
    return s


def match_footpaths(s: pd.DataFrame) -> pd.DataFrame:
    to_mga = Transformer.from_crs("EPSG:4326", "EPSG:7855", always_xy=True)
    fc = json.loads((RAW / "footpath_steepness.geojson").read_text("utf-8"))["features"]
    props = pd.DataFrame([f["properties"] for f in fc])
    geoms = np.array([shapely.geometry.shape(f["geometry"]) if f["geometry"] else None for f in fc], dtype=object)
    valid = (pd.notna(geoms) & props.gradepc.notna() & props.rlmin.notna() & props.rlmax.notna()
             & (props.distance >= MIN_SECTION_M)).to_numpy()
    props, geoms = props[valid].reset_index(drop=True), geoms[valid]
    geoms = shapely.transform(geoms, lambda xy: np.column_stack(to_mga.transform(xy[:, 0], xy[:, 1])))
    tree = shapely.STRtree(geoms)
    qa["footpath_sections_total"] = len(fc)
    qa["footpath_sections_used"] = int(valid.sum())

    x, y = to_mga.transform(s.longitude.to_numpy(), s.latitude.to_numpy())
    pts = shapely.points(x, y)
    (pi, gi), dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    near = props.iloc[gi].reset_index(drop=True)

    local = []
    for p in pts:
        idx = tree.query(p, predicate="dwithin", distance=LOCAL_SLOPE_RADIUS_M)
        sec = props.iloc[idx]
        local.append(np.average(sec.gradepc, weights=sec.distance) if len(sec) else np.nan)

    out = s.copy()
    out["easting_m"], out["northing_m"] = x, y
    order = np.argsort(pi)
    out["slope_pct"] = near.gradepc.to_numpy()[order]
    out["elevation_m"] = ((near.rlmin + near.rlmax) / 2).to_numpy()[order]
    out["slope_pct_local_mean"] = local
    out["footpath_match_dist_m"] = dist[order]
    out["footpath_mcc_id"] = near.mcc_id.to_numpy()[order]
    out["footpath_address"] = near.address.to_numpy()[order]
    return out


def dem_height(lon: float, lat: float, cache: dict) -> float:
    key = f"{lon:.7f},{lat:.7f}"
    if key not in cache:
        params = {
            "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "sr": 4326, "tolerance": 0,
            "mapExtent": f"{lon - .01},{lat - .01},{lon + .01},{lat + .01}", "imageDisplay": "400,400,96",
            "layers": "all", "returnGeometry": "false", "f": "json",
        }
        r = requests.get(GA_DEM_IDENTIFY, params=params, timeout=60,
                         headers={"User-Agent": "Mozilla/5.0 (pcs-preprocessing)"})
        r.raise_for_status()
        cache[key] = float(r.json()["results"][0]["attributes"]["Stretch.Pixel Value"])
    return cache[key]


def fill_from_dem(s: pd.DataFrame) -> pd.DataFrame:
    gap = s.footpath_match_dist_m > MAX_MATCH_M
    s["place_source"] = np.where(gap, "ga_dem_5m", "footpath_steepness")
    if not gap.any():
        return s
    to_wgs = Transformer.from_crs("EPSG:7855", "EPSG:4326", always_xy=True)
    cache = json.loads(DEM_CACHE.read_text()) if DEM_CACHE.exists() else {}
    d = DEM_STEP_M
    for i in s.index[gap]:
        x, y = s.at[i, "easting_m"], s.at[i, "northing_m"]
        z = {k: dem_height(*to_wgs.transform(x + dx, y + dy), cache)
             for k, dx, dy in [("c", 0, 0), ("e", d, 0), ("w", -d, 0), ("n", 0, d), ("s", 0, -d)]}
        slope = float(np.hypot(z["e"] - z["w"], z["n"] - z["s"]) / (2 * d) * 100)
        s.loc[i, ["elevation_m", "slope_pct", "slope_pct_local_mean"]] = [z["c"], slope, slope]
    DEM_CACHE.write_text(json.dumps(cache, indent=2))
    return s


def build_sensor_table(location_ids) -> pd.DataFrame:
    s = sensor_coordinates(location_ids)
    s = fill_from_dem(match_footpaths(s.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)))
    qa["sensors"] = len(s)
    qa["sensors_by_coord_source"] = s.coord_source.value_counts().to_dict()
    qa["sensors_filled_from_ga_dem"] = s.loc[s.place_source.eq("ga_dem_5m"), "location_id"].tolist()
    qa["sensors_slope_over_10pct"] = s.loc[s.slope_pct > 10, ["location_id", "slope_pct"]].values.tolist()
    qa["slope_pct_summary"] = s.slope_pct.describe().round(2).to_dict()
    cols = ["location_id", "sensor_description", "sensor_name", "location_type", "status", "installation_date",
            "latitude", "longitude", "easting_m", "northing_m", "coord_source", "slope_pct",
            "slope_pct_local_mean", "elevation_m", "place_source", "footpath_match_dist_m", "footpath_mcc_id",
            "footpath_address"]
    return s[cols].sort_values("location_id").reset_index(drop=True)


# ----------------------------------------------------------------- hours ----
def build_hour_table() -> pd.DataFrame:
    frames = [pd.DataFrame(json.loads(p.read_text())["hourly"])
              for p in sorted((RAW / "open_meteo_era5").glob("era5_hourly_*.json"))]
    w = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    w.index = pd.to_datetime(w.pop("time")).dt.tz_localize("UTC").rename("timestamp_utc")
    w = w.sort_index()
    assert w.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all(), "weather hours not contiguous"
    w[list(WEATHER_PRECEDING)] = w[list(WEATHER_PRECEDING)].shift(-1)
    w = w.rename(columns={**WEATHER_INSTANT, **WEATHER_PRECEDING})
    w = w.astype("float32").astype({"is_day": "Int8", "weather_code": "Int16"})
    qa["weather_hours"] = len(w)
    qa["weather_last_non_null_utc"] = str(w.temp_c.last_valid_index())

    mc = pd.read_parquet(RAW / "microclimate_sensors_data.parquet",
                         columns=["device_id", "received_at", "sensorlocation", *MC_VARS])
    mc = mc[~mc.sensorlocation.str.contains("rooftop", case=False)]
    for col, (_, lo, hi) in MC_VARS.items():
        mc[col] = mc[col].where(mc[col].between(lo, hi))
    mc["timestamp_utc"] = mc.received_at.dt.floor("h").astype("datetime64[ns, UTC]")
    per_device = mc.groupby(["timestamp_utc", "device_id"])[list(MC_VARS)].mean()
    hourly = per_device.groupby(level="timestamp_utc").median().astype("float32")
    hourly.columns = [MC_VARS[c][0] for c in hourly.columns]
    hourly["mc_n_devices"] = per_device.groupby(level="timestamp_utc").size().astype("Int8")
    qa["microclimate_devices_used"] = sorted(mc.device_id.unique().tolist())
    qa["microclimate_hours"] = len(hourly)

    w.index = w.index.astype("datetime64[ns, UTC]")
    h = w.join(hourly, how="outer")
    return h.reset_index()


# ------------------------------------------------------------------ join ----
def write_joined(c: pd.DataFrame, sensors: pd.DataFrame, hours: pd.DataFrame) -> int:
    place = sensors[["location_id", "sensor_description", "latitude", "longitude", "easting_m", "northing_m",
                     "slope_pct", "slope_pct_local_mean", "elevation_m", "place_source"]]
    # Directions add up to the target, so they stay out of the training table.
    base = c[["location_id", "timestamp_utc", "timestamp_local", "sensing_date", "hourday",
              "pedestriancount", "source", "suspect_outage_zero"]]
    out = OUT / "pcs_hourly_joined.parquet"
    writer, schema, n = None, None, 0
    for _, part in base.groupby(base.timestamp_utc.dt.year, sort=True):
        j = part.merge(place, on="location_id", how="left").merge(hours, on="timestamp_utc", how="left")
        assert len(j) == len(part), "join changed the row count"
        j = j.sort_values(["timestamp_utc", "location_id"])
        j["sensing_date"] = j.sensing_date.dt.date
        j["source"] = j.source.astype(str)
        t = pa.Table.from_pandas(j, preserve_index=False)
        if writer is None:
            schema = t.schema
            writer = pq.ParquetWriter(out, schema, compression="zstd")
        writer.write_table(t.cast(schema))
        n += len(j)
    writer.close()
    return n


def validate(path: Path, n_clean: int) -> None:
    t = pq.read_table(path, columns=["location_id", "timestamp_utc", "hourday", "sensing_date", "pedestriancount",
                                      "slope_pct", "elevation_m", "temp_c", "is_day"]).to_pandas()
    checks = {
        "joined_rows_equal_clean_rows": len(t) == n_clean,
        "no_duplicate_sensor_hours": not t.duplicated(["location_id", "timestamp_utc"]).any(),
        "no_negative_counts": bool((t.pedestriancount >= 0).all()),
        "every_row_has_slope_and_elevation": bool(t.slope_pct.notna().all() and t.elevation_m.notna().all()),
        "direction_sum_matches_total": qa["direction_sum_mismatches"] == 0,
    }
    month = pd.to_datetime(t.sensing_date).dt.month
    summer = month.isin([12, 1, 2])
    aft = t.temp_c[summer & t.hourday.between(14, 16)].mean()
    night = t.temp_c[summer & t.hourday.between(2, 4)].mean()
    checks["summer_afternoon_warmer_than_night"] = bool(aft > night + 3)
    qa["summer_afternoon_vs_night_temp_c"] = [round(float(aft), 2), round(float(night), 2)]
    checks["daylight_at_local_noon"] = bool(t.is_day[t.hourday == 12].dropna().eq(1).mean() > 0.999)
    checks["dark_at_local_midnight"] = bool(t.is_day[t.hourday == 0].dropna().eq(0).mean() > 0.999)
    qa["rows_without_weather"] = int(t.temp_c.isna().sum())
    qa["checks"] = checks
    qa["final_rows"] = len(t)
    qa["final_sensors"] = int(t.location_id.nunique())
    qa["final_range_utc"] = [str(t.timestamp_utc.min()), str(t.timestamp_utc.max())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors-only", action="store_true")
    args = ap.parse_args()
    INTERIM.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.sensors_only:
        ids = pd.read_parquet(RAW / "pcs_live_counts_20261016.parquet", columns=["location_id"]).location_id.unique()
        ids = sorted(set(ids) | set(range(1, 90)))
        print(build_sensor_table(ids).to_string())
        print(json.dumps(qa, indent=2, default=str))
        return

    counts = unify_counts()
    print("unified", len(counts))
    sensors = build_sensor_table(counts.location_id.unique())
    sensors.to_parquet(OUT / "sensor_table.parquet", index=False)
    counts = clean_counts(counts, sensors)
    counts.to_parquet(INTERIM / "counts_clean.parquet", index=False)
    missing_hours(counts).to_parquet(INTERIM / "missing_hours.parquet", index=False)
    print("clean", len(counts))

    hours = build_hour_table()
    hours.to_parquet(OUT / "hour_table.parquet", index=False)
    n = write_joined(counts, sensors, hours)
    print("joined", n)
    validate(OUT / "pcs_hourly_joined.parquet", len(counts))
    (OUT / "qa_report.json").write_text(json.dumps(qa, indent=2, default=str))
    print(json.dumps(qa, indent=2, default=str))
    failed = [k for k, ok in qa["checks"].items() if not ok]
    if failed:
        raise SystemExit(f"checks failed: {failed}")


if __name__ == "__main__":
    main()
