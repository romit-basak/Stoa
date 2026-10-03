"""Terrain for a site: GA 5 m LiDAR DEM clip (primary) plus 30 m global tiles for comparison.

Usage: python -m src.ingest.dem [--site melbourne]

FABDEM is deliberately absent (CC BY-NC-SA). Vicmap 1 m / Greater Melbourne LiDAR 2017-18 are
licensed or CC BY-NC, so they are not pulled either. See docs/data_cards/dem.md.
"""

from __future__ import annotations

import argparse

from src.ingest.common import download, load_config, raw_dir, record

SOURCE = "dem"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    args = ap.parse_args()

    cfg = load_config()
    dem = cfg["dem"]
    lon_min, lat_min, lon_max, lat_max = cfg["sites"][args.site]["bbox"]
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
