"""Hourly historical weather for a site from the Open-Meteo archive API (ERA5 reanalysis), one file per year.

Usage: python -m src.ingest.weather [--site melbourne]
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta

from src.ingest.common import download, load_config, raw_dir, record

SOURCE = "weather"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    args = ap.parse_args()

    cfg = load_config()
    w = cfg["weather"]
    site = cfg["sites"][args.site]
    out = raw_dir(cfg, SOURCE)

    start = date.fromisoformat(site.get("weather_start", w["start_date"]))
    # The archive lags real time by a few days.
    end = datetime.now(UTC).date() - timedelta(days=7)
    for year in range(start.year, end.year + 1):
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
