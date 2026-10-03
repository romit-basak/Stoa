"""Download ESA WorldCover 10 m tiles and clip to the site bbox.

Usage: python -m src.ingest.worldcover [--site melbourne]
"""

from __future__ import annotations

import argparse

import rasterio
from rasterio.windows import from_bounds

from src.ingest.common import download, load_config, raw_dir, record

SOURCE = "worldcover"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="melbourne")
    args = ap.parse_args()

    cfg = load_config()
    wc = cfg["worldcover"]
    bbox = cfg["sites"][args.site]["bbox"]
    out = raw_dir(cfg, SOURCE)

    for tile in wc["tiles"][args.site]:
        url = wc["url_template"].format(tile=tile)
        tile_path = out / url.rsplit("/", 1)[-1]
        download(url, tile_path)
        record(out, tile_path, url, license="CC-BY-4.0")

        clip_path = out / f"{args.site}_worldcover_2021.tif"
        with rasterio.open(tile_path) as src:
            window = from_bounds(*bbox, transform=src.transform)
            profile = src.profile | {
                "height": int(window.height),
                "width": int(window.width),
                "transform": src.window_transform(window),
                "compress": "deflate",
            }
            with rasterio.open(clip_path, "w", **profile) as dst:
                dst.write(src.read(window=window))
        record(
            out,
            clip_path,
            url,
            derived_from=tile_path.name,
            bbox=bbox,
            license="CC-BY-4.0",
        )


if __name__ == "__main__":
    main()
