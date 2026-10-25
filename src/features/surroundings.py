"""What surrounds each segment: land cover, tree canopy, terrain slope, jobs and businesses.

Portable (any city has the inputs):
- wc_tree_share, wc_grass_share, wc_built_share: share of ESA WorldCover pixels of each class
  in a window of about `landcover_radius_m` around the segment (a 10 m raster, so too coarse
  for the path itself; it describes the setting);
- dem_slope_mean_pct, dem_slope_max_pct: the segment's grade from the site's DEM, sampled every
  `slope_step_m` along it (and measured over at least that distance), so DEMs of different
  resolution (GA 5 m, USGS 1 m or 10 m) are measured the same way. Empty on bridges and in
  tunnels, where a bare-earth DEM doesn't describe the walkway, and off the DEM's coverage
  (its nodata value, or the site's `dem_nodata`: the GA LiDAR tile fills uncovered ground
  with 0 and declares no nodata).

Melbourne-rich (City of Melbourne layers; empty at other sites):
- canopy_cover: share of a `canopy_buffer_m` buffer under tree canopy (2021 urban forest);
- clue_jobs_<r>: jobs in CLUE blocks (latest census year) whose centre is within r metres along
  the network; businesses_<r>: business establishments (latest year) within r metres.
Both are one snapshot applied to the whole count record, like the OSM extract.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from scipy.ndimage import uniform_filter
from scipy.sparse import csr_matrix

from src.ingest.common import REPO_ROOT

WORLDCOVER = {"tree": 10, "grass": 30, "built": 50}


def _points_along(
    edges: gpd.GeoDataFrame, step_m: float, min_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points every step_m along each segment, ends included; segment row and distance of each."""
    lengths = edges.length.to_numpy()
    n = np.maximum(min_points, np.ceil(lengths / step_m).astype(int) + 1)
    rows = np.repeat(np.arange(len(edges)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    k = np.arange(len(rows)) - start
    frac = k / np.repeat(n - 1, n)
    pts = shapely.line_interpolate_point(
        edges.geometry.to_numpy()[rows], frac, normalized=True
    )
    return pts, rows, frac * lengths[rows]


def _sample(
    path,
    pts: np.ndarray,
    crs: int,
    array: np.ndarray | None = None,
    nodata: float | None = None,
) -> np.ndarray:
    """Raster values (band 1, or `array` on the raster's grid) at points given in `crs`.

    NaN outside the raster and where band 1 holds the raster's nodata or `nodata`.
    """
    with rasterio.open(path) as src:
        tf = Transformer.from_crs(crs, src.crs, always_xy=True)
        x, y = tf.transform(shapely.get_x(pts), shapely.get_y(pts))
        rows, cols = rasterio.transform.rowcol(src.transform, x, y)
        rows, cols = np.asarray(rows), np.asarray(cols)
        data = src.read(1) if array is None else array
        missing = [v for v in (src.nodata, nodata) if v is not None]
    inside = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    out = np.full(len(pts), np.nan)
    out[inside] = data[rows[inside], cols[inside]]
    if missing and array is None:
        out[np.isin(out, missing)] = np.nan
    return out


def _pixel_m(src) -> float:
    """Pixel size in metres (geometric mean of its sides, at the raster's centre)."""
    if not src.crs.is_geographic:
        return float(np.sqrt(abs(src.res[0] * src.res[1])))
    lat = src.xy(src.height // 2, src.width // 2)[1]
    dy = abs(src.res[1]) * 111_320
    dx = abs(src.res[0]) * 111_320 * np.cos(np.radians(lat))
    return float(np.sqrt(dx * dy))


def landcover_shares(edges: gpd.GeoDataFrame, path, radius_m: float) -> pd.DataFrame:
    """Share of each WorldCover class in a window around each segment's points."""
    out = pd.DataFrame({"segment_id": edges["segment_id"].to_numpy()})
    if path is None or not path.exists():
        for name in WORLDCOVER:
            out[f"wc_{name}_share"] = np.nan
        return out
    with rasterio.open(path) as src:
        lc = src.read(1)
        px_m = _pixel_m(src)
    # An odd window about 2 x radius_m wide, centred on the pixel.
    size = max(1, round(2 * radius_m / px_m) | 1)
    pts, rows, _ = _points_along(edges, 10.0, 1)
    for name, cls in WORLDCOVER.items():
        focal = uniform_filter(
            (lc == cls).astype(np.float32), size=size, mode="nearest"
        )
        v = _sample(path, pts, edges.crs.to_epsg(), focal)
        out[f"wc_{name}_share"] = pd.Series(v).groupby(rows).mean().to_numpy()
    return out


def segment_slope(
    edges: gpd.GeoDataFrame, path, step_m: float, nodata: float | None = None
) -> pd.DataFrame:
    """Mean and steepest grade along each segment, in percent, from the DEM."""
    out = pd.DataFrame({"segment_id": edges["segment_id"].to_numpy()})
    if path is None or not path.exists():
        out["dem_slope_mean_pct"] = out["dem_slope_max_pct"] = np.nan
        return out
    pts, rows, dist = _points_along(edges, step_m, 2)
    z = _sample(path, pts, edges.crs.to_epsg(), nodata=nodata)
    same = rows[1:] == rows[:-1]
    # Rise over at least step_m: on a 2 m segment, two samples either side of a DEM pixel edge
    # (a wall, a kerb) would otherwise read as a cliff.
    grade = np.abs(np.diff(z)) / np.maximum(np.diff(dist), step_m) * 100
    g = pd.Series(grade[same]).groupby(rows[1:][same])
    out["dem_slope_mean_pct"] = g.mean().reindex(range(len(edges))).to_numpy()
    out["dem_slope_max_pct"] = g.max().reindex(range(len(edges))).to_numpy()
    # A bare-earth DEM gives the ground under a bridge or above a tunnel, not the walkway.
    raised = np.zeros(len(edges), dtype=bool)
    for tag in ("bridge", "tunnel"):
        if tag in edges:
            raised |= edges[tag].notna().to_numpy() & edges[tag].ne("no").to_numpy()
    out.loc[raised, ["dem_slope_mean_pct", "dem_slope_max_pct"]] = np.nan
    return out


def polygon_cover(
    edges: gpd.GeoDataFrame, polys: gpd.GeoDataFrame | None, buffer_m: float
) -> np.ndarray:
    """Share of each segment's buffer covered by the polygons (NaN when there are none)."""
    if polys is None:
        return np.full(len(edges), np.nan)
    buf = gpd.GeoDataFrame(
        {"i": np.arange(len(edges))}, geometry=edges.buffer(buffer_m), crs=edges.crs
    )
    p = polys[["geometry"]].to_crs(edges.crs).reset_index(drop=True)
    p["geometry"] = p.geometry.make_valid()
    pairs = gpd.sjoin(buf, p, predicate="intersects")
    inter = shapely.intersection(
        pairs.geometry.to_numpy(),
        p.geometry.to_numpy()[pairs["index_right"].to_numpy()],
    )
    area = pd.Series(shapely.area(inter)).groupby(pairs["i"].to_numpy()).sum()
    return (
        (area.reindex(range(len(edges)), fill_value=0) / buf.area).clip(0, 1).to_numpy()
    )


def weighted_reach(
    edges,
    nodes,
    points: gpd.GeoDataFrame,
    weights,
    radii: list[int],
    csr,
    snap_m: float,
) -> dict[int, np.ndarray]:
    """Sum of point weights within each network radius of every segment."""
    from src.features.segment_features import _reach_pairs as reach_pairs
    from src.features.segment_features import snap_to_nodes as snap

    indptr, nbr, eid, wt, eu, ev = csr
    s = snap(points.to_crs(edges.crs), nodes, snap_m).to_numpy()
    keep = s >= 0
    s, w = s[keep], np.asarray(weights, dtype=float)[keep]
    src, idx = np.unique(s, return_inverse=True)
    src_w = np.bincount(idx, weights=w)
    ps, pe, pd_ = reach_pairs(
        src.astype(np.int64), indptr, nbr, eid, wt, eu, ev, float(max(radii))
    )
    out = {}
    for r in radii:
        sel = pd_ <= r
        reach = csr_matrix(
            (np.ones(sel.sum()), (pe[sel], ps[sel])), (len(edges), len(src))
        )
        out[r] = reach @ src_w
    return out


def surroundings(
    edges, nodes, site_name: str, data_cfg: dict[str, Any], fcfg: dict[str, Any], csr
) -> pd.DataFrame:
    cfg = fcfg["surroundings"]
    raw = REPO_ROOT / data_cfg["raw_dir"]
    files = cfg.get("sites", {}).get(site_name, {})
    path = lambda key: raw / files[key] if key in files else None
    out = landcover_shares(edges, path("worldcover"), cfg["landcover_radius_m"])
    out = out.merge(
        segment_slope(edges, path("dem"), cfg["slope_step_m"], files.get("dem_nodata")),
        on="segment_id",
    )

    canopy = (
        gpd.read_file(path("canopy"))
        if path("canopy") and path("canopy").exists()
        else None
    )
    out["canopy_cover"] = polygon_cover(edges, canopy, cfg["canopy_buffer_m"])

    radii_jobs, radii_biz = cfg["jobs_radii_m"], cfg["business_radii_m"]
    if path("clue_blocks") and path("clue_blocks").exists():
        blocks = gpd.read_file(path("clue_blocks"))
        jobs = pd.read_parquet(
            path("clue_jobs"),
            columns=["census_year", "block_id", "total_jobs_in_block"],
        )
        jobs = jobs[jobs["census_year"] == jobs["census_year"].max()]
        b = blocks.merge(jobs, on="block_id")
        b["geometry"] = b.geometry.representative_point()
        jobs_w = b["total_jobs_in_block"].fillna(0)
        reach = weighted_reach(edges, nodes, b, jobs_w, radii_jobs, csr, cfg["snap_m"])
        for r in radii_jobs:
            out[f"clue_jobs_{r}"] = reach[r]
    else:
        for r in radii_jobs:
            out[f"clue_jobs_{r}"] = np.nan
    if path("businesses") and path("businesses").exists():
        biz = pd.read_parquet(
            path("businesses"), columns=["census_year", "longitude", "latitude"]
        )
        biz = biz[biz["census_year"] == biz["census_year"].max()].dropna()
        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(biz["longitude"], biz["latitude"]), crs=4326
        )
        ones = np.ones(len(pts))
        reach = weighted_reach(edges, nodes, pts, ones, radii_biz, csr, cfg["snap_m"])
        for r in radii_biz:
            out[f"businesses_{r}"] = reach[r]
    else:
        for r in radii_biz:
            out[f"businesses_{r}"] = np.nan
    return out
