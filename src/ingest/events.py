"""AFL games (venue, local start time, crowd not included) from the Squiggle API, one file per year.

Usage: python -m src.ingest.events [--all]

Fetches the years not held yet and the current one (daily, in the stoa_pipeline task
`pull_events`); --all fetches every year again.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from src.ingest.common import (
    download,
    load_config,
    raw_dir,
    record,
    setup_logging,
    years_to_fetch,
)

SOURCE = "events"


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="fetch every year again")
    args = ap.parse_args()
    cfg = load_config()
    ev = cfg["events"]
    out = raw_dir(cfg, SOURCE)
    held = {int(p.stem.rsplit("_", 1)[1]) for p in out.glob("afl_games_*.json")}
    now = datetime.now(UTC).year
    for year in years_to_fetch(ev["start_year"], now, held, args.all):
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
