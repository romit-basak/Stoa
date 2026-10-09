"""Match the City of Melbourne count sensors to OSM graph segments.

Each sensor gets up to N candidate segments within the search radius. The pick is the nearest
after two adjustments: a road centreline whose footpaths are mapped separately
(sidewalk=separate) is skipped, because the sensor stands on one of those footpaths, and
crossings get a small penalty, because sensors sit on footpaths beside them. Hand-checked
choices in configs/sensor_overrides.yaml replace the automatic pick.

Outputs in data/processed/features/<site>/:
- sensor_segment_map.parquet: location_id -> segment_id, with distance, match method and flags
- sensor_match_review.csv: the candidates for each flagged sensor, for a person to check
- sensor_osm_features.parquet: location_id + every feature of the matched segment. This is the
  table to join onto the counts (key: location_id).
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

from src.features.common import Site, log, write_meta, write_parquet
from src.ingest.common import REPO_ROOT

CROSSING_PENALTY_M = 5.0
# Sensor direction labels -> the axis (degrees from north, 0-180) a matching segment runs along.
DIRECTION_AXIS = {"north": 0.0, "south": 0.0, "east": 90.0, "west": 90.0}


def sensor_axis(d1: str | None, d2: str | None) -> float | None:
    for d in (d1, d2):
        if isinstance(d, str) and d.strip().lower() in DIRECTION_AXIS:
            return DIRECTION_AXIS[d.strip().lower()]
    return None


def axis_diff(a: float, b: float) -> float:
    """Smallest angle between two undirected axes (0-90)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def candidates(
    sensors: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Every (sensor, segment) pair within the search radius, ranked by adjusted distance."""
    r = cfg["search_radius_m"]
    s_idx, e_idx = edges.sindex.query(sensors.geometry, predicate="dwithin", distance=r)
    pairs = pd.DataFrame(
        {
            "location_id": sensors["location_id"].to_numpy()[s_idx],
            "segment_id": edges["segment_id"].to_numpy()[e_idx],
            "distance_m": shapely.distance(
                edges.geometry.to_numpy()[e_idx], sensors.geometry.to_numpy()[s_idx]
            ),
            "highway": edges["highway"].to_numpy()[e_idx],
            "name": edges["name"].to_numpy()[e_idx],
            "sidewalk_state": edges["sidewalk_state"].to_numpy()[e_idx],
            "is_crossing": edges["is_crossing"].to_numpy()[e_idx],
            "bearing_deg": edges["bearing_deg"].to_numpy()[e_idx],
        }
    )
    preferred = set(cfg["preferred_highway"])
    pairs["footpath_class"] = pairs["highway"].isin(preferred)
    pairs["skip"] = pairs["sidewalk_state"].eq("separate")
    pairs["score"] = pairs["distance_m"] + np.where(
        pairs["is_crossing"], CROSSING_PENALTY_M, 0.0
    )
    pairs = pairs[~pairs["skip"]].sort_values(["location_id", "score"])
    pairs["rank"] = pairs.groupby("location_id").cumcount() + 1
    return pairs[pairs["rank"] <= cfg["candidates"]].drop(columns="skip")


def match_sensors(
    sensors: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    cfg: dict[str, Any],
    overrides: dict[int, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (map, candidates). The map has one row per sensor, matched or not."""
    overrides = overrides or {}
    cand = candidates(sensors, edges, cfg)
    first = cand[cand["rank"] == 1].set_index("location_id")
    second = cand[cand["rank"] == 2].set_index("location_id")
    rows = []
    for _, s in sensors.iterrows():
        lid = int(s["location_id"])
        axis = sensor_axis(s.get("direction_1"), s.get("direction_2"))
        row = {
            "location_id": lid,
            "sensor_description": s.get("sensor_description"),
            "location_type": s.get("location_type"),
            "segment_id": None,
            "distance_m": np.nan,
            "method": "none",
            "runner_up_gap_m": np.nan,
            "bearing_agrees": None,
        }
        if lid in overrides:
            seg = overrides[lid]
            geom = edges.loc[edges["segment_id"] == seg, "geometry"]
            row.update(
                segment_id=seg,
                distance_m=float(geom.distance(s.geometry).iloc[0])
                if len(geom)
                else np.nan,
                method="override",
            )
        elif lid in first.index:
            f = first.loc[lid]
            row.update(
                segment_id=f["segment_id"], distance_m=f["distance_m"], method="auto"
            )
            if lid in second.index:
                row["runner_up_gap_m"] = second.loc[lid, "score"] - f["score"]
            if axis is not None:
                row["bearing_agrees"] = axis_diff(f["bearing_deg"], axis) <= 30.0
            row["matched_highway"] = f["highway"]
            row["matched_name"] = f["name"]
        rows.append(row)
    m = pd.DataFrame(rows)
    reasons = []
    for _, r in m.iterrows():
        why = []
        if r["method"] == "none":
            why.append("no segment in range")
        elif r["method"] == "auto":
            if r["distance_m"] > cfg["review_if_nearest_over_m"]:
                why.append("far")
            if r["runner_up_gap_m"] <= cfg["review_if_runner_up_within_m"]:
                why.append("close runner-up")
            if r.get("matched_highway") not in cfg["preferred_highway"]:
                why.append("road centreline")
            if r["bearing_agrees"] is not None and not bool(r["bearing_agrees"]):
                why.append("bearing disagrees")
        if r["location_type"] != "Outdoor":
            why.append(f"{str(r['location_type']).lower()} sensor")
        reasons.append("; ".join(why))
    m["review_reasons"] = reasons
    m["needs_review"] = m["review_reasons"].ne("") & m["method"].ne("override")
    return m, cand


def load_overrides(site: Site, cfg: dict[str, Any]) -> dict[int, str]:
    path = REPO_ROOT / cfg["overrides_file"]
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return {int(k): str(v) for k, v in (data.get(site.name) or {}).items()}


def sensor_match(
    site: Site, fcfg: dict[str, Any], data_cfg: dict[str, Any]
) -> dict[str, Any]:
    cfg = fcfg["sensor_match"]
    sensors = gpd.read_file(
        REPO_ROOT
        / data_cfg["raw_dir"]
        / "melbourne_portal"
        / "pedestrian-counting-system-sensor-locations.geojson"
    ).to_crs(site.crs)
    edges = gpd.read_parquet(site.graph_dir / "edges.parquet")
    m, cand = match_sensors(sensors, edges, cfg, load_overrides(site, cfg))
    write_parquet(m, site.processed / "sensor_segment_map.parquet")

    review = m.loc[
        m["needs_review"], ["location_id", "sensor_description", "review_reasons"]
    ].merge(cand, on="location_id", how="left")
    review["chosen_segment_id"] = ""
    review_path = site.processed / "sensor_match_review.csv"
    review.to_csv(review_path, index=False)
    log.info(
        "wrote %s (%d sensors to check)",
        review_path.relative_to(REPO_ROOT),
        m["needs_review"].sum(),
    )

    feats = pd.read_parquet(site.processed / "osm_segment_features.parquet").drop(
        columns="geometry"
    )
    keep = [
        "location_id",
        "sensor_description",
        "location_type",
        "segment_id",
        "distance_m",
        "method",
        "needs_review",
    ]
    joined = m[keep].rename(
        columns={"distance_m": "match_distance_m", "method": "match_method"}
    )
    joined = joined.merge(feats, on="segment_id", how="left", validate="many_to_one")
    write_parquet(joined, site.processed / "sensor_osm_features.parquet")

    d = m["distance_m"]
    stats = {
        "sensors": len(m),
        "matched": int(m["segment_id"].notna().sum()),
        "within_5m": int((d <= 5).sum()),
        "within_15m": int((d <= 15).sum()),
        "needs_review": int(m["needs_review"].sum()),
        "overrides": int((m["method"] == "override").sum()),
    }
    write_meta(site.processed / "sensor_match_meta.json", site=site.name, **stats)
    return stats
