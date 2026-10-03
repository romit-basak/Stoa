"""AFL games (venue, local start time, crowd not included) from the Squiggle API, one file per year.

Usage: python -m src.ingest.events
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.ingest.common import download, load_config, raw_dir, record

SOURCE = "events"


def main() -> None:
    cfg = load_config()
    ev = cfg["events"]
    out = raw_dir(cfg, SOURCE)
    for year in range(ev["start_year"], datetime.now(UTC).year + 1):
        params = {"q": "games", "year": year}
        dest = out / f"afl_games_{year}.json"
        download(ev["squiggle_url"], dest, params=params)
        record(
            out,
            dest,
            ev["squiggle_url"],
            params=params,
            license="unclear (public fixture facts)",
        )


if __name__ == "__main__":
    main()
