"""Building shade on each segment, for each position of the sun the site sees.

The sun's position is binned (azimuth in 30-degree steps, elevation in bands from 5 degrees
up). For each bin the site's sun reaches, every building footprint is swept away from the sun
by height / tan(elevation) into a shadow polygon, and points every `step_m` along each segment
are tested against the shadows. A segment's shaded fraction is the share of its points in
shadow. Below 5 degrees the sun is treated as down: fully shaded, with no row in the table.

Two variants, because heights come from different places:
- portable: OSM buildings; height from `height`, else `building:levels` x level_height_m,
  else default_height_m (a labelled assumption, counted in the meta);
- rich: the city's own heights where a site has them (City of Melbourne 2023 footprints,
  City of Boston buildings), empty elsewhere.

The sweep uses each footprint's convex hull with its moved copy, which slightly overstates
the shadow of a concave building (an L-shaped block also shades its inner corner).

Writes data/processed/features/<site>/osm_segment_shade.parquet: segment_id, sun_bin,
shade_portable, shade_rich (rows only where some variant is above zero; absent means 0) and
shade_meta.json, which holds the bin edges the training frame uses to look up each hour.
"""

from __future__ import annotations

import re
import time
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from src.features.common import Site, log, write_meta, write_parquet
from src.features.sun import solar_position
from src.ingest.common import REPO_ROOT

# --------------------------------------------------------------------------------------------
# Sun bins
# --------------------------------------------------------------------------------------------


def bin_of(
    elevation: np.ndarray, azimuth: np.ndarray, cfg: dict[str, Any]
) -> np.ndarray:
    """Sun bin for each position; -1 where the sun is below the lowest elevation edge."""
    el_edges = np.asarray(cfg["elevation_edges_deg"], dtype=float)
    n_el = len(el_edges) - 1
    az_i = (np.asarray(azimuth, dtype=float) // cfg["azimuth_step_deg"]).astype(int)
    az_i %= int(360 // cfg["azimuth_step_deg"])
    el_i = (
        np.searchsorted(el_edges, np.asarray(elevation, dtype=float), side="right") - 1
    )
    up = (el_i >= 0) & (el_i < n_el)
    return np.where(up, az_i * n_el + np.clip(el_i, 0, n_el - 1), -1)


def bin_centre(b: int, cfg: dict[str, Any]) -> tuple[float, float]:
    """(azimuth, elevation) at the middle of bin b."""
    el_edges = cfg["elevation_edges_deg"]
    n_el = len(el_edges) - 1
    az_i, el_i = divmod(int(b), n_el)
    az = (az_i + 0.5) * cfg["azimuth_step_deg"]
    el = 0.5 * (el_edges[el_i] + el_edges[el_i + 1])
    return az, el


def reachable_bins(lat: float, lon: float, cfg: dict[str, Any]) -> list[int]:
    """Bins the sun passes through during a year at this place (every 10 minutes of 2025)."""
    t = pd.Series(
        pd.date_range("2025-01-01", "2025-12-31 23:50", freq="10min", tz="UTC")
    )
    elevation, azimuth = solar_position(t, lat, lon)
    b = bin_of(elevation, azimuth, cfg)
    return sorted(int(x) for x in np.unique(b[b >= 0]))


# --------------------------------------------------------------------------------------------
# Heights
# --------------------------------------------------------------------------------------------


def _height_m(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(m.group(1)) if m else None


def osm_heights(buildings: gpd.GeoDataFrame, cfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """OSM footprints with a height: tagged, else from levels, else the labelled default."""
    h = (
        buildings["height"].map(_height_m)
        if "height" in buildings
        else pd.Series(np.nan)
    )
    levels = (
        buildings["building_levels"].map(_height_m)
        if "building_levels" in buildings
        else pd.Series(np.nan, index=buildings.index)
    )
    from_levels = levels * cfg["level_height_m"]
    height = h.fillna(from_levels).fillna(cfg["default_height_m"]).astype(float)
    source = np.where(
        h.notna(), "tag", np.where(from_levels.notna(), "levels", "default")
    )
    return gpd.GeoDataFrame(
        {"height_m": height.to_numpy(), "height_source": source},
        geometry=buildings.geometry.to_numpy(),
        crs=buildings.crs,
    )


def city_heights(
    site: Site, data_cfg: dict[str, Any], cfg: dict[str, Any]
) -> gpd.GeoDataFrame | None:
    """The site's own building heights, if it has a source (`shade.city_heights`)."""
    spec = cfg.get("city_heights", {}).get(site.name)
    if not spec:
        return None
    path = REPO_ROOT / data_cfg["raw_dir"] / spec["path"]
    if not path.exists():
        log.warning("%s not held: no rich shade for %s", path, site.name)
        return None
    b = gpd.read_file(path)
    height = pd.to_numeric(b[spec["height_field"]], errors="coerce") * spec.get(
        "scale", 1.0
    )
    out = gpd.GeoDataFrame(
        {"height_m": height.to_numpy(), "height_source": "city"},
        geometry=b.geometry.to_numpy(),
        crs=b.crs,
    ).to_crs(site.crs)
    return out[out["height_m"] > 0].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Shadows
# --------------------------------------------------------------------------------------------


def shadows(
    buildings: gpd.GeoDataFrame, azimuth: float, elevation: float
) -> np.ndarray:
    """Shadow polygon of each footprint for the sun at (azimuth, elevation), in metres."""
    length = buildings["height_m"].to_numpy() / np.tan(np.radians(elevation))
    # The shadow falls away from the sun: azimuth is clockwise from north, y points north.
    dx = -np.sin(np.radians(azimuth)) * length
    dy = -np.cos(np.radians(azimuth)) * length
    geoms = shapely.make_valid(buildings.geometry.to_numpy())
    moved = np.array(
        [shapely.affinity.translate(g, x, y) for g, x, y in zip(geoms, dx, dy)],
        dtype=object,
    )
    return shapely.convex_hull(shapely.union(geoms, moved))


def sample_points(
    edges: gpd.GeoDataFrame, step_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Points every step_m along each segment (at least its midpoint); and their segment rows."""
    lengths = edges.length.to_numpy()
    n = np.maximum(1, np.floor(lengths / step_m).astype(int))
    rows = np.repeat(np.arange(len(edges)), n)
    # Fractions at the middle of each of the n equal pieces of a segment.
    start = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(len(rows)) - start
    frac = (k + 0.5) / np.repeat(n, n)
    pts = shapely.line_interpolate_point(
        edges.geometry.to_numpy()[rows], frac, normalized=True
    )
    return pts, rows


def shaded_fraction(
    points: np.ndarray, rows: np.ndarray, n_rows: int, shadow: np.ndarray
) -> np.ndarray:
    """Share of each segment's points that lie in any shadow."""
    tree = STRtree(shadow)
    hit = np.unique(tree.query(points, predicate="intersects")[0])
    shaded = np.bincount(rows[hit], minlength=n_rows)
    total = np.bincount(rows, minlength=n_rows)
    return shaded / np.maximum(total, 1)


def shade_table(
    edges: gpd.GeoDataFrame,
    variants: dict[str, gpd.GeoDataFrame | None],
    bins: list[int],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    points, rows = sample_points(edges, cfg["step_m"])
    ids = edges["segment_id"].to_numpy()
    parts = []
    for b in bins:
        az, el = bin_centre(b, cfg)
        cols = {}
        for name, buildings in variants.items():
            if buildings is None or buildings.empty:
                cols[name] = np.full(len(edges), np.nan)
                continue
            cols[name] = shaded_fraction(
                points, rows, len(edges), shadows(buildings, az, el)
            )
        df = pd.DataFrame({"segment_id": ids, "sun_bin": np.int16(b), **cols})
        keep = np.zeros(len(df), dtype=bool)
        for name in variants:
            keep |= df[name].fillna(0).to_numpy() > 0
        parts.append(df[keep])
    out = pd.concat(parts, ignore_index=True)
    for name in variants:
        out[f"shade_{name}"] = out.pop(name).astype("float32")
    return out


def shade(site: Site, fcfg: dict[str, Any], data_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = fcfg["shade"]
    t = time.perf_counter()
    edges = gpd.read_parquet(site.graph_dir / "edges.parquet")
    osm = osm_heights(gpd.read_parquet(site.interim / "buildings.parquet"), cfg)
    variants = {"portable": osm, "rich": city_heights(site, data_cfg, cfg)}
    lon, lat = data_cfg["sites"][site.name]["center"]
    bins = reachable_bins(lat, lon, cfg)
    table = shade_table(edges, variants, bins, cfg)
    write_parquet(table, site.processed / "osm_segment_shade.parquet")
    stats = {
        "segments": len(edges),
        "n_bins": len(bins),
        "rows": len(table),
        "osm_buildings": len(osm),
        "osm_height_from": osm["height_source"].value_counts().to_dict(),
        "city_buildings": 0 if variants["rich"] is None else len(variants["rich"]),
        "runtime_s": round(time.perf_counter() - t, 1),
    }
    write_meta(
        site.processed / "shade_meta.json",
        site=site.name,
        bins=bins,
        azimuth_step_deg=cfg["azimuth_step_deg"],
        elevation_edges_deg=cfg["elevation_edges_deg"],
        step_m=cfg["step_m"],
        **stats,
    )
    log.info("shade: %s", stats)
    return stats
