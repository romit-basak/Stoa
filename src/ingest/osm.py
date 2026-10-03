"""Download a weekly OpenStreetMap extract (.osm.pbf) covering a site.

Usage: python -m src.ingest.osm [--site melbourne]

Overpass times out on a full-metadata dump of a dense CBD, so we use pre-cut BBBike city
extracts. Clipping to the site bbox happens in src/features, not here.
"""

from __future__ import annotations

import argparse

import requests

from src.ingest.common import USER_AGENT, download, load_config, raw_dir, record

SOURCE = "osm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    args = ap.parse_args()

    cfg = load_config()
    url = cfg["osm"]["extracts"][args.site]
    out = raw_dir(cfg, SOURCE)
    dest = out / f"{args.site}.osm.pbf"

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
