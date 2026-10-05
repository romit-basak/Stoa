"""Aerial imagery for path segmentation at a US campus site.

Usage: python -m src.ingest.imagery [--site neu_boston]

- NAIP (public domain): latest year on Microsoft Planetary Computer, read by window from the
  cloud-optimised GeoTIFF and clipped to the site bbox (a full quarter-quad is ~1.8 GB).
- MassGIS 2025 orthos (15 cm leaf-off): the 1500 m JPEG 2000 tiles that intersect the bbox,
  found via the published tile index.
"""

from __future__ import annotations

import argparse
import zipfile

import geopandas as gpd
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box

from src.ingest.common import USER_AGENT, download, load_config, raw_dir, record

SOURCE = "imagery"


def latest_naip(cfg: dict, bbox: list[float]) -> dict:
    r = requests.post(
        cfg["stac_search"],
        json={"collections": ["naip"], "bbox": bbox, "limit": 100},
        headers={"User-Agent": USER_AGENT},
        timeout=120,
    )
    r.raise_for_status()
    items = r.json()["features"]
    footprint = box(*bbox)
    covering = [i for i in items if footprint.within(box(*i["bbox"]))]
    if not covering:
        raise RuntimeError(
            "no single NAIP item covers the bbox; mosaicking not implemented"
        )
    return max(covering, key=lambda i: i["properties"]["datetime"])


def naip(cfg: dict, site: str, bbox: list[float], out) -> None:
    item = latest_naip(cfg, bbox)
    token = requests.get(cfg["sas_token"], timeout=60).json()["token"]
    href = item["assets"]["image"]["href"]
    dest = out / f"{site}_naip_{item['properties']['datetime'][:4]}.tif"
    with rasterio.open(f"/vsicurl/{href}?{token}") as src:
        window = from_bounds(
            *transform_bounds(4326, src.crs, *bbox), transform=src.transform
        )
        window = window.round_offsets().round_lengths()
        profile = src.profile | {
            "height": window.height,
            "width": window.width,
            "transform": src.window_transform(window),
            "compress": "deflate",
            "tiled": True,
        }
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(src.read(window=window))
    record(
        out,
        dest,
        href,
        stac_item=item["id"],
        acquired=item["properties"]["datetime"],
        gsd_m=item["properties"].get("gsd"),
        bbox=bbox,
        license="public domain (USDA FSA NAIP)",
    )


def massgis(cfg: dict, site: str, bbox: list[float], out) -> None:
    index_zip = out / "massgis_coq2025_index.zip"
    download(cfg["index"], index_zip)
    record(out, index_zip, cfg["index"])
    index = gpd.read_file(f"zip://{index_zip}")
    footprint = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(index.crs).iloc[0]
    tiles = index[index.intersects(footprint)]
    url_col = next(c for c in index.columns if c.upper() == "URL")
    for url in tiles[url_col]:
        dest = out / url.rsplit("/", 1)[-1]
        download(url, dest, timeout=1800)
        with zipfile.ZipFile(dest) as z:
            members = z.namelist()
        record(
            out,
            dest,
            url,
            members=members,
            license="public domain (MassGIS FAQ); credit MassGIS (Bureau of Geographic Information), Commonwealth of Massachusetts EOTSS",
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="neu_boston")
    args = ap.parse_args()

    cfg = load_config()
    bbox = cfg["sites"][args.site]["bbox"]
    out = raw_dir(cfg, SOURCE)
    naip(cfg["imagery"]["naip"], args.site, bbox, out)
    massgis(cfg["imagery"]["massgis"], args.site, bbox, out)


if __name__ == "__main__":
    main()
