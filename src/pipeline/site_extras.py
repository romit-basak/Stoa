"""Pull the layers a site has beyond OSM, DEM, land cover, weather and calendars.

Usage: python -m src.pipeline.site_extras --site SITE

Each name in `sites.<site>.extra_ingest` (configs/data.yaml) is an ingest module run with
`--site SITE`: for NEU Boston, the city's building heights and the aerial imagery. A site
with no extras does nothing, so the stoa_site_build DAG can run this task for every site.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.ingest.common import REPO_ROOT, load_config, log, setup_logging


def commands(site: str, cfg: dict) -> list[list[str]]:
    extras = cfg["sites"][site].get("extra_ingest", [])
    return [
        [sys.executable, "-m", f"src.ingest.{name}", "--site", site] for name in extras
    ]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    args = ap.parse_args()
    todo = commands(args.site, load_config())
    if not todo:
        log.info("%s has no extra layers", args.site)
    for cmd in todo:
        log.info("running %s", " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
