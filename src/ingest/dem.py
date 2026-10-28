"""Terrain for a site: GA 5 m LiDAR DEM clip (primary) plus 30 m global tiles for comparison.

Usage: python -m src.ingest.dem [--site melbourne|neu_boston]

US sites get a USGS 3DEP 1 m clip from the ImageServer instead.

FABDEM is deliberately absent (CC BY-NC-SA). Vicmap 1 m / Greater Melbourne LiDAR 2017-18 are
licensed or CC BY-NC, so they are not pulled either. See docs/data_cards/dem.md.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.merge import merge

from src.ingest.common import download, load_config, raw_dir, record, setup_logging

SOURCE = "dem"


def tiles(
    xmin: float, ymin: float, width: int, height: int, res: float, max_px: int
) -> list[tuple[float, float, int, int, float]]:
    """(xmin, ymin, width, height, res) of each tile, none wider or taller than max_px."""
    out = []
    for y0 in range(0, height, max_px):
        for x0 in range(0, width, max_px):
            w, h = min(max_px, width - x0), min(max_px, height - y0)
            out.append((xmin + x0 * res, ymin + y0 * res, w, h, res))
    return out


def usgs_params(
    xmin: float, ymin: float, width: int, height: int, res: float, epsg: int
) -> dict:
    """ImageServer export of a clip whose pixels are exactly res metres."""
    return {
        "bbox": f"{xmin},{ymin},{xmin + width * res},{ymin + height * res}",
        "bboxSR": epsg,
        "imageSR": epsg,
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }


def mosaic(paths: list[Path], dest: Path) -> None:
    """Merge tiles that share a grid into one GeoTIFF."""
    sources = [rasterio.open(p) for p in paths]
    try:
        data, transform = merge(sources)
        meta = sources[0].meta | {
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
        }
    finally:
        for s in sources:
            s.close()
    with rasterio.open(dest, "w", **meta) as dst:
        dst.write(data)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    args = ap.parse_args()

    cfg = load_config()
    dem = cfg["dem"]
    site = cfg["sites"][args.site]
    lon_min, lat_min, lon_max, lat_max = site["bbox"]
    out = raw_dir(cfg, SOURCE)

    source = site.get("dem", {}).get("source")
    if source == "ga_5m":
        res = dem["ga_5m_res_deg"]
        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": "1",
            "crs": "EPSG:4283",
            "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "resx": res,
            "resy": res,
            "format": "GeoTIFF",
        }
        dest = out / f"{args.site}_ga_lidar_5m.tif"
        download(dem["ga_5m_wcs"], dest, params=params)
        record(out, dest, dem["ga_5m_wcs"], params=params, license="CC-BY-4.0")
    elif source == "usgs_3dep":
        # USGS 3DEP clip in the site's UTM zone, sized so pixels are exactly res metres.
        epsg = site["utm_epsg"]
        to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
        xmin, ymin, xmax, ymax = to_utm.transform_bounds(
            lon_min, lat_min, lon_max, lat_max
        )
        res = site["dem"]["res_m"]
        width, height = math.ceil((xmax - xmin) / res), math.ceil((ymax - ymin) / res)
        dest = out / f"{args.site}_usgs_3dep_{res:g}m.tif"
        parts = tiles(xmin, ymin, width, height, res, dem.get("usgs_max_px", 2500))
        params = usgs_params(xmin, ymin, width, height, res, epsg)
        if len(parts) == 1:
            download(dem["usgs_3dep_imageserver"], dest, params=params)
        else:
            # The service refuses big exports (HTTP 500), so a large clip comes in tiles.
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for i, part in enumerate(parts):
                    path = Path(tmp) / f"tile_{i}.tif"
                    download(
                        dem["usgs_3dep_imageserver"],
                        path,
                        params=usgs_params(*part, epsg),
                    )
                    paths.append(path)
                mosaic(paths, dest)
            params = {"tiles": len(parts), **params}
        record(
            out,
            dest,
            dem["usgs_3dep_imageserver"],
            params=params,
            license="public domain (USGS)",
        )
    else:
        raise SystemExit(
            f"sites.{args.site}.dem.source in configs/data.yaml is {source!r}: "
            "set it to ga_5m (Australia) or usgs_3dep (US)"
        )

    for name, url in dem["global_tiles"].get(args.site, {}).items():
        dest = (
            out / url.rsplit("/", 1)[-1]
        )  # keep tile names; GDAL's SRTM driver needs them
        download(url, dest)
        lic = (
            "Copernicus DEM licence (free, attribution)"
            if "copernicus" in name
            else "public domain"
        )
        record(out, dest, url, license=lic)


if __name__ == "__main__":
    main()
