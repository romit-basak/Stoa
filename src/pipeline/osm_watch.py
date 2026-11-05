"""Compare the pinned OSM extract with the latest one and report path changes near sensors.

Usage: python -m src.pipeline.osm_watch [--site melbourne]

The graph is built from a pinned extract (data/raw/osm/<site>.osm.pbf) so segment IDs stay
stable, but the map keeps changing: a new footpath or a closed laneway near a sensor changes
what the sensor counts without any change in our data. Once a week (the stoa_osm_watch DAG)
this downloads the latest extract to a temporary folder, reads the walkable ways from both, and
reports ways added, removed, re-routed (different nodes) or retagged (a kept tag changed)
within `radius_m` of a count sensor. The pinned file is never replaced; refreshing it stays a
deliberate step (stoa_pipeline with refresh_osm, then re-check the sensor overrides).

Changes near sensors are a warning, emailed like validation warnings. Writes
data/processed/validation/<site>/osm_changes.json and osm_changes.csv.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.features.common import load_features_config, site_settings
from src.features.osm_extract import read_walkable_ways
from src.features.sensor_match import load_sensor_locations
from src.ingest.common import REPO_ROOT, download, load_config, log, setup_logging
from src.pipeline.validate import WARNING, Check, send_warning_alert

RADIUS_M = 100.0


def way_changes(
    old: gpd.GeoDataFrame, new: gpd.GeoDataFrame, tags: list[str]
) -> gpd.GeoDataFrame:
    """One row per way added, removed, re-routed or retagged between two extracts."""
    o, n = old.set_index("way_id"), new.set_index("way_id")
    rows = []
    for way_id in n.index.difference(o.index):
        rows.append(
            {"way_id": way_id, "change": "added", "geometry": n.at[way_id, "geometry"]}
        )
    for way_id in o.index.difference(n.index):
        rows.append(
            {
                "way_id": way_id,
                "change": "removed",
                "geometry": o.at[way_id, "geometry"],
            }
        )
    for way_id in o.index.intersection(n.index):
        a, b = o.loc[way_id], n.loc[way_id]
        if list(a["node_ids"]) != list(b["node_ids"]):
            rows.append(
                {"way_id": way_id, "change": "rerouted", "geometry": b["geometry"]}
            )
            continue
        changed = [t for t in tags if t in o.columns and _differs(a.get(t), b.get(t))]
        if changed:
            rows.append(
                {
                    "way_id": way_id,
                    "change": "retagged",
                    "tags": ",".join(changed),
                    "geometry": b["geometry"],
                }
            )
    cols = ["way_id", "change", "tags", "geometry"]
    if not rows:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=old.crs)
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=old.crs).reindex(columns=cols)
    return out.sort_values(["change", "way_id"], ignore_index=True)


def _differs(a: Any, b: Any) -> bool:
    return not (pd.isna(a) and pd.isna(b)) and a != b


def near_sensors(
    changes: gpd.GeoDataFrame, sensors: gpd.GeoDataFrame, crs: int, radius_m: float
) -> pd.DataFrame:
    """Every change within radius_m of a sensor, one row per (sensor, way)."""
    cols = ["location_id", "way_id", "change", "tags", "distance_m"]
    if changes.empty:
        return pd.DataFrame(columns=cols)
    c = changes.to_crs(crs)
    s = sensors[["location_id", "geometry"]].drop_duplicates("location_id").to_crs(crs)
    zones = s.assign(geometry=s.buffer(radius_m))
    joined = gpd.sjoin(zones, c, how="inner", predicate="intersects")
    points = s.set_index("location_id").geometry
    joined["distance_m"] = [
        points[lid].distance(c.geometry.iloc[i])
        for lid, i in zip(joined["location_id"], joined["index_right"])
    ]
    return joined[cols].sort_values(["location_id", "way_id"], ignore_index=True)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="default: the hourly table's site")
    args = ap.parse_args()
    data_cfg, fcfg = load_config(), load_features_config()
    site = site_settings(args.site or fcfg["counts_hourly"]["site"], data_cfg, fcfg)
    url = data_cfg["osm"]["extracts"][site.name]
    with tempfile.TemporaryDirectory() as tmp:
        latest = Path(tmp) / f"{site.name}_latest.osm.pbf"
        download(url, latest, timeout=1800)
        new = read_walkable_ways(latest, site.bbox, fcfg)
    old = read_walkable_ways(site.pbf, site.bbox, fcfg)
    changes = way_changes(old, new, fcfg["way_tags"])
    sensors = load_sensor_locations(data_cfg)
    near = near_sensors(changes, sensors, site.crs, RADIUS_M)
    log.info(
        "%d way changes since the pinned extract, %d within %.0f m of a sensor",
        len(changes),
        len(near),
        RADIUS_M,
    )
    out = REPO_ROOT / fcfg["processed_dir"] / "validation" / site.name
    out.mkdir(parents=True, exist_ok=True)
    near.to_csv(out / "osm_changes.csv", index=False)
    summary = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "site": site.name,
        "pinned": site.pbf.name,
        "latest_url": url,
        "changes": changes["change"].value_counts().to_dict(),
        "near_sensors": len(near),
        "sensors_affected": int(near["location_id"].nunique()) if len(near) else 0,
    }
    (out / "osm_changes.json").write_text(json.dumps(summary, indent=2) + "\n")
    if len(near):
        check = Check(
            "osm: paths changed near sensors",
            WARNING,
            False,
            f"{len(near)} way changes within {RADIUS_M:.0f} m of "
            f"{summary['sensors_affected']} sensors since the pinned extract",
            summary,
        )
        send_warning_alert(
            [check], f"data/processed/validation/{site.name}/osm_changes.csv"
        )


if __name__ == "__main__":
    main()
