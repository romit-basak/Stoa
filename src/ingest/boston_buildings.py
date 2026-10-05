"""Boston building footprints with heights (Buildings with Roof Breaks) for a site bbox.

Usage: python -m src.ingest.boston_buildings [--site neu_boston]

Pages the ArcGIS FeatureServer and checks the result against the server's own count.
Heights (GRND_ELEV_2010, ROOF_ELEV_2010, BLDG_HGT_2010) are US feet. License: PDDL.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import requests

from src.ingest.common import USER_AGENT, load_config, raw_dir, record

SOURCE = "boston_buildings"


def query(url: str, params: dict[str, Any]) -> dict[str, Any]:
    r = requests.get(
        f"{url}/query", params=params, headers={"User-Agent": USER_AGENT}, timeout=300
    )
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="neu_boston")
    args = ap.parse_args()

    cfg = load_config()
    bb = cfg["boston_buildings"]
    bbox = cfg["sites"][args.site]["bbox"]
    out = raw_dir(cfg, SOURCE)

    where = {
        "where": "1=1",
        "geometry": ",".join(map(str, bbox)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
    }
    expected = query(
        bb["feature_server"], {**where, "returnCountOnly": "true", "f": "json"}
    )["count"]

    # Offset paging on this server repeats and skips rows, so fetch every ID first and then
    # request the features in ID batches.
    ids = sorted(
        query(bb["feature_server"], {**where, "returnIdsOnly": "true", "f": "json"})[
            "objectIds"
        ]
    )
    features: list[dict[str, Any]] = []
    for i in range(0, len(ids), bb["page_size"]):
        batch = ids[i : i + bb["page_size"]]
        page = query(
            bb["feature_server"],
            {
                "objectIds": ",".join(map(str, batch)),
                "outFields": "*",
                "outSR": 4326,
                "f": "geojson",
            },
        )
        features += page["features"]

    dest = out / f"{args.site}_buildings.geojson"
    dest.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    record(
        out,
        dest,
        bb["feature_server"],
        bbox=bbox,
        features=len(features),
        server_count=expected,
        license="PDDL-1.0",
    )
    if len(features) != expected:
        print(f"WARNING: got {len(features)} features, server reports {expected}")


if __name__ == "__main__":
    main()
