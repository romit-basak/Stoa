"""Terrain for a site: GA 5 m LiDAR DEM clip (primary) plus 30 m global tiles for comparison.

Usage: python -m src.ingest.dem [--site melbourne|neu_boston]

US sites get a USGS 3DEP 1 m clip from the ImageServer instead.

FABDEM is deliberately absent (CC BY-NC-SA). Vicmap 1 m / Greater Melbourne LiDAR 2017-18 are
licensed or CC BY-NC, so they are not pulled either. See docs/data_cards/dem.md.
"""

from __future__ import annotations

import argparse
import math

from pyproj import Transformer

from src.ingest.common import download, load_config, raw_dir, record

SOURCE = "dem"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    args = ap.parse_args()

    cfg = load_config()
    dem = cfg["dem"]
    site = cfg["sites"][args.site]
    lon_min, lat_min, lon_max, lat_max = site["bbox"]
    out = raw_dir(cfg, SOURCE)

    if args.site == "melbourne":
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
    elif "utm_epsg" in site:
        # USGS 3DEP 1 m clip in the site's UTM zone, sized so pixels are exactly res metres.
        epsg = site["utm_epsg"]
        to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
        xmin, ymin, xmax, ymax = to_utm.transform_bounds(
            lon_min, lat_min, lon_max, lat_max
        )
        res = dem["usgs_3dep_res_m"]
        width, height = math.ceil((xmax - xmin) / res), math.ceil((ymax - ymin) / res)
        params = {
            "bbox": f"{xmin},{ymin},{xmin + width * res},{ymin + height * res}",
            "bboxSR": epsg,
            "imageSR": epsg,
            "size": f"{width},{height}",
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_BilinearInterpolation",
            "f": "image",
        }
        dest = out / f"{args.site}_usgs_3dep_1m.tif"
        download(dem["usgs_3dep_imageserver"], dest, params=params)
        record(
            out,
            dest,
            dem["usgs_3dep_imageserver"],
            params=params,
            license="public domain (USGS)",
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
