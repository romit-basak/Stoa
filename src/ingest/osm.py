"""Download a weekly OpenStreetMap extract (.osm.pbf) covering a site.

Usage: python -m src.ingest.osm [--site melbourne] [--refresh]

Overpass times out on a full-metadata dump of a dense CBD, so we use pre-cut BBBike city
extracts. Clipping to the site bbox happens in src/features, not here.

The extract is pinned: once a site's file exists it is kept, and a new one is downloaded only
with --refresh. Graph segment IDs (and the hand-checked sensor matches that use them) belong to
one extract, so replacing it is a deliberate step, not something a scheduled run does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from src.ingest.common import (
    USER_AGENT,
    download,
    load_config,
    log,
    raw_dir,
    record,
    setup_logging,
)

SOURCE = "osm"


def should_download(dest: Path, refresh: bool) -> bool:
    return refresh or not dest.exists()


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="replace the pinned extract with the latest one (changes segment IDs)",
    )
    args = ap.parse_args()

    cfg = load_config()
    url = cfg["osm"]["extracts"][args.site]
    out = raw_dir(cfg, SOURCE)
    dest = out / f"{args.site}.osm.pbf"

    if not should_download(dest, args.refresh):
        log.info("keeping pinned extract %s (use --refresh to replace it)", dest.name)
        return
    head = requests.head(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    download(url, dest, timeout=1800)
    record(
        out,
        dest,
        url,
        extract_last_modified=head.headers.get("Last-Modified"),
        license="ODbL-1.0",
    )


if __name__ == "__main__":
    main()
