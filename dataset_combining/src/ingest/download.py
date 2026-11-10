"""Acquire every raw source into data/raw/ and write a manifest entry per file.

Raw files are never edited. Live PCS counts are saved as a dated snapshot so a
later pull adds a file instead of overwriting the rows already held (the portal
drops data older than about two years).

Usage:  python -I src/ingest/download.py [--only NAME ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"

CKAN = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
CC_BY = "CC BY 4.0 (City of Melbourne Open Data)"
TODAY = dt.date.today()

# One point in central Melbourne (Town Hall); ERA5 snaps it to its 0.25 deg cell.
MELB_LAT, MELB_LON = -37.8136, 144.9631
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "rain", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
    "wind_direction_10m", "shortwave_radiation", "sunshine_duration",
    "surface_pressure", "weather_code", "is_day",
]
# Archive starts 2009-05-01 00:00 Melbourne time = 2009-04-30 14:00 UTC.
WEATHER_START = dt.date(2009, 4, 30)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream(url: str, dest: Path, params: dict | None = None, retries: int = 4) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, params=params, stream=True, timeout=(30, 900)) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except requests.RequestException as e:
            wait = 15 * (attempt + 1)
            print(f"  {dest.name}: {e} - retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}")


def record(manifest: dict, path: Path, url: str, licence: str, **extra) -> None:
    manifest[path.name] = {
        "url": url,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "licence": licence,
        **extra,
    }
    print(f"  saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def portal_export(manifest: dict, dataset: str, fmt: str, dest: Path) -> None:
    url = f"{CKAN}/{dataset}/exports/{fmt}"
    print(f"{dataset} -> {dest.name}")
    stream(url, dest)
    meta = requests.get(f"{CKAN}/{dataset}", timeout=60).json()["metas"]["default"]
    record(manifest, dest, url, CC_BY, portal_records_count=meta.get("records_count"),
           portal_data_processed=meta.get("data_processed"))


def get_pcs_live(m):
    portal_export(m, "pedestrian-counting-system-monthly-counts-per-hour", "parquet",
                  RAW / f"pcs_live_counts_{TODAY:%Y%m%d}.parquet")


def get_pcs_archive(m):
    ds = "pedestrian-counting-system-monthly-counts-per-hour"
    att = "pedestrian_counting_system_monthly_counts_per_hour_may_2009_to_14_dec_2022_csv_zip"
    url = f"{CKAN}/{ds}/attachments/{att}"
    dest = RAW / "pcs_archive_2009_2022.csv.zip"
    print(f"archive -> {dest.name}")
    stream(url, dest)
    record(m, dest, url, CC_BY)


def get_sensor_locations(m):
    portal_export(m, "pedestrian-counting-system-sensor-locations", "parquet",
                  RAW / "pcs_sensor_locations.parquet")


def get_sensor_locations_2022(m):
    # The current table omits 18 retired archive sensors. The Oct 2022 Socrata
    # copy of the same City of Melbourne table (status R/I/A) still lists them.
    src = "https://data.melbourne.vic.gov.au/resource/h57g-5234"
    url = f"http://web.archive.org/web/20221024034400id_/{src}"
    dest = RAW / "pcs_sensor_locations_2022-10-24_wayback.json"
    print(f"wayback sensor locations -> {dest.name}")
    stream(url, dest)
    record(m, dest, url, CC_BY, original_url=src, snapshot="2022-10-24T03:44:00Z")


PCS_WAYBACK_SNAPSHOTS = ["20250502183548"]


def get_pcs_wayback(m):
    # The live window rolls forward, so Nov 2022 - Oct 2024 is no longer on the
    # portal. Wayback holds full portal CSV exports made while it still was.
    # 20250502183548 covers 2021-07-01..2025-05-02 and matches the archive and
    # live rows it overlaps on ~98.5% of hours (rest differ by a few people).
    src = ("https://data.melbourne.vic.gov.au/explore/dataset/"
           "pedestrian-counting-system-monthly-counts-per-hour/download?format=csv")
    for ts in PCS_WAYBACK_SNAPSHOTS:
        url = f"http://web.archive.org/web/{ts}id_/{src}"
        dest = RAW / f"pcs_counts_wayback_{ts}.csv"
        print(f"wayback counts {ts} -> {dest.name}")
        stream(url, dest)
        record(m, dest, url, CC_BY, original_url=src, snapshot=ts)


def get_footpath_steepness(m):
    # GeoJSON keeps the footpath polygons needed for the point-in-section match.
    portal_export(m, "footpath-steepness", "geojson", RAW / "footpath_steepness.geojson")


def get_microclimate(m):
    portal_export(m, "microclimate-sensors-data", "parquet", RAW / "microclimate_sensors_data.parquet")


def get_weather(m):
    url = "https://archive-api.open-meteo.com/v1/archive"
    out_dir = RAW / "open_meteo_era5"
    out_dir.mkdir(exist_ok=True)
    for year in range(WEATHER_START.year, TODAY.year + 1):
        start = max(dt.date(year, 1, 1), WEATHER_START)
        end = min(dt.date(year, 12, 31), TODAY)
        dest = out_dir / f"era5_hourly_{year}.json"
        params = {
            "latitude": MELB_LAT, "longitude": MELB_LON,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": ",".join(WEATHER_VARS), "models": "era5",
            "timezone": "GMT", "wind_speed_unit": "kmh",
        }
        print(f"open-meteo {year} -> {dest.name}")
        for attempt in range(6):
            r = requests.get(url, params=params, timeout=(30, 300))
            if r.status_code == 429:
                time.sleep(65)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError(f"open-meteo kept rate limiting for {year}")
        dest.write_bytes(r.content)
        record(m, dest, r.url, "CC BY 4.0 (Open-Meteo; ERA5 Copernicus/ECMWF)",
               hours=len(r.json()["hourly"]["time"]))
        time.sleep(8)  # stay well under the per-minute weighted call limit


SOURCES = {
    "pcs_live": get_pcs_live,
    "pcs_archive": get_pcs_archive,
    "pcs_wayback": get_pcs_wayback,
    "sensor_locations": get_sensor_locations,
    "sensor_locations_2022": get_sensor_locations_2022,
    "footpath_steepness": get_footpath_steepness,
    "weather": get_weather,
    "microclimate": get_microclimate,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(SOURCES))
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    failed = []
    for name in args.only or SOURCES:
        try:
            SOURCES[name](manifest)
        except Exception as e:  # keep the last good copy and report
            print(f"!! {name} failed: {e}", file=sys.stderr)
            failed.append(name)
        MANIFEST.write_text(json.dumps(manifest, indent=2))
    if failed:
        sys.exit(f"sources failed: {failed}")


if __name__ == "__main__":
    main()
