"""Hourly historical weather for a site from the Open-Meteo archive API (ERA5 reanalysis), one file per year.

Usage: python -m src.ingest.weather [--site melbourne] [--full]

ERA5 for past years doesn't change, so by default only missing years and the last two are
fetched (last year's file may still end short of 31 December if it was fetched in early
January). --full re-fetches every year.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta

from src.ingest.common import download, load_config, raw_dir, record, setup_logging

SOURCE = "weather"


def years_to_fetch(
    start: date, end: date, existing: set[int], full: bool = False
) -> list[int]:
    """Years to download: all with --full, otherwise missing years plus the last two."""
    years = range(start.year, end.year + 1)
    if full:
        return list(years)
    return [y for y in years if y not in existing or y >= end.year - 1]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    ap.add_argument("--full", action="store_true", help="re-fetch every year")
    args = ap.parse_args()

    cfg = load_config()
    w = cfg["weather"]
    site = cfg["sites"][args.site]
    out = raw_dir(cfg, SOURCE)

    start = date.fromisoformat(site.get("weather_start", w["start_date"]))
    # The archive lags real time by a few days.
    end = datetime.now(UTC).date() - timedelta(days=7)
    existing = {
        int(p.stem.rsplit("_", 1)[1]) for p in out.glob(f"{args.site}_openmeteo_*.json")
    }
    for year in years_to_fetch(start, end, existing, args.full):
        params = {
            "latitude": site["center"][1],
            "longitude": site["center"][0],
            "start_date": max(start, date(year, 1, 1)).isoformat(),
            "end_date": min(end, date(year, 12, 31)).isoformat(),
            "hourly": ",".join(w["hourly"]),
            "timezone": site["timezone"],
        }
        dest = out / f"{args.site}_openmeteo_{year}.json"
        download(w["url"], dest, params=params)
        record(
            out,
            dest,
            w["url"],
            params=params,
            license="CC-BY-4.0 (data); API free tier non-commercial",
        )


if __name__ == "__main__":
    main()
