"""Run the OSM feature pipeline for a site: extract -> graph -> features -> sensors.

Usage: python -m src.features.osm [--site melbourne] [--only extract,graph,features,sensors]

Each stage reads the previous stage's files, so stages can be re-run on their own (and wrapped
as separate Airflow tasks later). Sensor matching needs the City of Melbourne sensor locations,
so it only runs for melbourne.
"""

from __future__ import annotations

import argparse
import logging
import time

from src.features.common import load_features_config, log, site_settings
from src.features.graph import graph
from src.features.osm_extract import extract
from src.features.segment_features import segment_features
from src.features.sensor_match import sensor_match
from src.ingest.common import load_config

STAGES = ["extract", "graph", "features", "sensors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    ap.add_argument("--only", help="comma-separated subset of: " + ", ".join(STAGES))
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    data_cfg = load_config()
    fcfg = load_features_config()
    site = site_settings(args.site, data_cfg, fcfg)
    if not site.pbf.exists():
        raise SystemExit(
            f"{site.pbf} not found: run python -m src.ingest.osm --site {site.name}"
        )
    stages = args.only.split(",") if args.only else STAGES
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")
    if "sensors" in stages and site.name != "melbourne":
        log.warning(
            "sensor matching uses the City of Melbourne sensors; skipped for %s",
            site.name,
        )
        stages = [s for s in stages if s != "sensors"]

    run = {
        "extract": lambda: extract(site, fcfg),
        "graph": lambda: graph(site, fcfg),
        "features": lambda: segment_features(site, fcfg),
        "sensors": lambda: sensor_match(site, fcfg, data_cfg),
    }
    for stage in STAGES:
        if stage in stages:
            t = time.perf_counter()
            result = run[stage]()
            log.info("%s done in %.0f s: %s", stage, time.perf_counter() - t, result)


if __name__ == "__main__":
    main()
