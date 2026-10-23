"""Run the OSM feature pipeline for a site: extract -> graph -> features -> sensors.

Usage: python -m src.features.osm [--site melbourne] [--only extract,graph,features,sensors]
                                  [--if-stale]

Each stage reads the previous stage's files, so stages can be re-run on their own (the Airflow
DAG runs them as separate tasks). Sensor matching needs the City of Melbourne sensor locations,
so it only runs for melbourne.

With --if-stale, a stage is skipped when its outputs exist and none of its inputs changed since
it last ran. A stage's inputs are the previous stage's, plus its own: the extract file, bbox and
CRS, features config and feature code (extract); the sensor locations file and sensor overrides
(sensors). Each successful stage writes that fingerprint to data/interim/osm/<site>/stamps/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.features.common import (
    Site,
    config_hash,
    load_features_config,
    log,
    site_settings,
)
from src.features.graph import graph
from src.features.osm_extract import extract
from src.features.segment_features import segment_features
from src.features.sensor_match import (
    retired_locations_path,
    sensor_locations_path,
    sensor_match,
)
from src.ingest.common import REPO_ROOT, load_config, setup_logging, sha256

STAGES = ["extract", "graph", "features", "sensors"]
FEATURE_CODE = REPO_ROOT / "src" / "features"


def _hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path) if path.exists() else "missing"


def stage_outputs(site: Site) -> dict[str, list[Path]]:
    p, g = site.processed, site.graph_dir
    return {
        "extract": [site.interim / f"{n}.parquet" for n in ("ways", "pois", "transit")],
        "graph": [g / "edges.parquet", g / "nodes.parquet"],
        "features": [
            p / "osm_segment_features.parquet",
            p / "osm_segment_open_hours.parquet",
        ],
        "sensors": [
            p / "sensor_segment_map.parquet",
            p / "sensor_osm_features.parquet",
        ],
    }


def fingerprints(
    site: Site, fcfg: dict[str, Any], data_cfg: dict[str, Any]
) -> dict[str, str]:
    """One hash per stage that changes whenever anything the stage depends on changes."""
    code = _hash(*(_file_hash(f) for f in sorted(FEATURE_CODE.glob("*.py"))))
    fp = {
        "extract": _hash(
            _file_hash(site.pbf), str(site.bbox), str(site.crs), config_hash(fcfg), code
        )
    }
    fp["graph"] = _hash(fp["extract"], "graph")
    fp["features"] = _hash(fp["graph"], "features")
    overrides = REPO_ROOT / fcfg["sensor_match"]["overrides_file"]
    fp["sensors"] = _hash(
        fp["features"],
        _file_hash(sensor_locations_path(data_cfg)),
        _file_hash(retired_locations_path(data_cfg)),
        _file_hash(overrides),
    )
    return fp


def is_current(site: Site, stage: str, fingerprint: str) -> bool:
    stamp = site.interim / "stamps" / f"{stage}.json"
    if not stamp.exists() or not all(p.exists() for p in stage_outputs(site)[stage]):
        return False
    return json.loads(stamp.read_text()).get("fingerprint") == fingerprint


def write_stamp(site: Site, stage: str, fingerprint: str) -> None:
    stamp = site.interim / "stamps" / f"{stage}.json"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        json.dumps({"fingerprint": fingerprint, "finished": time.time()}) + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    ap.add_argument("--only", help="comma-separated subset of: " + ", ".join(STAGES))
    ap.add_argument(
        "--if-stale",
        action="store_true",
        help="skip stages whose inputs haven't changed since their last successful run",
    )
    args = ap.parse_args()
    setup_logging()

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
    fps = fingerprints(site, fcfg, data_cfg)
    for stage in STAGES:
        if stage not in stages:
            continue
        if args.if_stale and is_current(site, stage, fps[stage]):
            log.info("%s is up to date; skipped", stage)
            continue
        t = time.perf_counter()
        result = run[stage]()
        write_stamp(site, stage, fps[stage])
        log.info("%s done in %.0f s: %s", stage, time.perf_counter() - t, result)


if __name__ == "__main__":
    main()
