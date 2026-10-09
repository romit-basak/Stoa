"""Shared helpers for the feature pipeline: config, site settings, paths and run metadata."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml

from src.ingest.common import REPO_ROOT, load_config

log = logging.getLogger("stoa.features")


def load_features_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or REPO_ROOT / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    """Short hash of a config dict, stored with outputs so a run can be traced to its settings."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass(frozen=True)
class Site:
    name: str
    # lon_min, lat_min, lon_max, lat_max (WGS84)
    bbox: tuple[float, float, float, float]
    crs: int  # projected CRS in metres
    timezone: str
    interim: Path  # data/interim/osm/<site>
    processed: Path  # data/processed/features/<site>
    pbf: Path

    @property
    def graph_dir(self) -> Path:
        return self.interim / "graph"


def site_settings(
    name: str,
    data_cfg: dict[str, Any] | None = None,
    feat_cfg: dict[str, Any] | None = None,
) -> Site:
    data_cfg = data_cfg or load_config()
    feat_cfg = feat_cfg or load_features_config()
    s = data_cfg["sites"][name]
    return Site(
        name=name,
        bbox=tuple(s["bbox"]),
        crs=feat_cfg["sites"][name]["crs"],
        timezone=s["timezone"],
        interim=REPO_ROOT / feat_cfg["interim_dir"] / "osm" / name,
        processed=REPO_ROOT / feat_cfg["processed_dir"] / "features" / name,
        pbf=REPO_ROOT / data_cfg["raw_dir"] / "osm" / f"{name}.osm.pbf",
    )


def write_parquet(gdf: Any, path: Path) -> Path:
    """Write a (Geo)DataFrame to Parquet, creating the folder. GeoDataFrames become GeoParquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(gdf, gpd.GeoDataFrame):
        gdf.to_parquet(path, index=False, schema_version="1.0.0")
    else:
        gdf.to_parquet(path, index=False)
    log.info("wrote %s (%d rows)", path.relative_to(REPO_ROOT), len(gdf))
    return path


def write_meta(path: Path, **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")


def pbf_sha256(site: Site) -> str | None:
    """SHA-256 of the site's extract as recorded by src.ingest.osm (None if not recorded)."""
    manifest = site.pbf.parent / "MANIFEST.json"
    if not manifest.exists():
        return None
    entry = json.loads(manifest.read_text()).get(site.pbf.name, {})
    return entry.get("sha256")


def tag_matcher(rules: list[str]):
    """Return f(tags) -> bool for rules like "amenity=cafe" or "shop=*"."""
    parsed = [r.split("=", 1) for r in rules]

    def match(tags: dict[str, str]) -> bool:
        for key, value in parsed:
            v = tags.get(key)
            if v is not None and (value == "*" or v == value):
                return True
        return False

    return match
