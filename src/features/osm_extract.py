"""Turn a site's .osm.pbf extract into GeoParquet layers the rest of the pipeline reads.

Walkable ways are read with pyosmium because the graph needs OSM node IDs (GDAL's line layer
drops them). POIs, buildings, green space, pedestrian areas and transit stops are read with
pyogrio, which assembles multipolygons for us. Everything is clipped to the site bbox and
projected to the site CRS.

Outputs in data/interim/osm/<site>/: ways, pois, buildings, green, ped_areas, transit (.parquet)
and extract_meta.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmium
import pandas as pd
import pyogrio
from shapely.geometry import LineString

from src.features.common import (
    Site,
    config_hash,
    log,
    pbf_sha256,
    tag_matcher,
    write_meta,
    write_parquet,
)

# pyogrio puts tags without their own column into an hstore string: "k"=>"v","k2"=>"v2".
_HSTORE = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')
_OGR_META = {"osm_id", "osm_way_id", "other_tags", "geometry"}


def parse_hstore(s: str | None) -> dict[str, str]:
    if not isinstance(s, str) or not s:
        return {}
    return {k.replace('\\"', '"'): v.replace('\\"', '"') for k, v in _HSTORE.findall(s)}


def ogr_tags(df: pd.DataFrame) -> list[dict[str, str]]:
    """Rebuild each row's full tag dict from pyogrio's named columns plus other_tags."""
    cols = [c for c in df.columns if c not in _OGR_META]
    records = df[cols].to_dict("records")
    out = []
    for rec, other in zip(records, df["other_tags"]):
        tags = {k: str(v) for k, v in rec.items() if v is not None and not pd.isna(v)}
        tags.update(parse_hstore(other))
        out.append(tags)
    return out


def is_walkable(tags: dict[str, str], wf: dict[str, Any]) -> bool:
    """OSMnx-style walk filter. An explicit foot=yes/designated/permissive overrides access=no."""
    hw = tags.get("highway")
    if hw is None or tags.get("area") == "yes":
        return False
    foot_ok = tags.get("foot") in wf["cycleway_foot_values"]
    if hw == "cycleway":
        return foot_ok
    if hw in wf["exclude_highway"]:
        return False
    for key, bad in wf["exclude_tag_values"].items():
        if tags.get(key) in bad and not (key == "access" and foot_ok):
            return False
    return True


def _in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def read_walkable_ways(pbf: Path, bbox, fcfg: dict[str, Any]) -> gpd.GeoDataFrame:
    """Walkable ways with at least one node inside bbox, with node IDs and kept tags (WGS84)."""
    wf, keep = fcfg["walk_filter"], fcfg["way_tags"]
    rows = []
    fp = (
        osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("highway"))
    )
    for w in fp:
        tags = dict(w.tags)
        if not is_walkable(tags, wf):
            continue
        refs, coords = [], []
        for n in w.nodes:
            if n.location.valid():
                refs.append(n.ref)
                coords.append((n.lon, n.lat))
        # Drop repeated consecutive nodes (bad mapping) so no segment has zero length.
        dedup = [
            (r, c)
            for i, (r, c) in enumerate(zip(refs, coords))
            if i == 0 or r != refs[i - 1]
        ]
        if len(dedup) < 2 or not any(_in_bbox(*c, bbox) for _, c in dedup):
            continue
        refs, coords = zip(*dedup)
        rows.append(
            {
                "way_id": w.id,
                "node_ids": list(refs),
                **{k: tags.get(k) for k in keep},
                "geometry": LineString(coords),
            }
        )
    if not rows:
        cols = ["way_id", "node_ids", *keep]
        return gpd.GeoDataFrame(columns=cols, geometry=gpd.GeoSeries([], crs=4326))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def _read_layer(
    pbf: Path, layer: str, bbox, where: str | None = None
) -> gpd.GeoDataFrame:
    df = pyogrio.read_dataframe(pbf, layer=layer, bbox=bbox, where=where)
    df["tags"] = ogr_tags(df)
    if layer == "multipolygons":
        # A polygon from a closed way has osm_way_id; one from a relation has osm_id.
        df["osm_type"] = df["osm_way_id"].notna().map({True: "way", False: "relation"})
        df["osm_id"] = df["osm_way_id"].fillna(df["osm_id"])
    else:
        df["osm_type"] = "node"
    df["osm_id"] = pd.to_numeric(df["osm_id"]).astype("int64")
    return df[["osm_type", "osm_id", "tags", "geometry"]]


def _select(df: gpd.GeoDataFrame, rules: list[str]) -> gpd.GeoDataFrame:
    m = tag_matcher(rules)
    return df[df["tags"].map(m)].copy()


def _finish(df: gpd.GeoDataFrame, crs: int, keep_tags: list[str]) -> gpd.GeoDataFrame:
    """Project, pull named tags into columns, keep the full tag dict as a JSON string."""
    df = df.to_crs(crs)
    for k in keep_tags:
        df[k.replace(":", "_")] = df["tags"].map(lambda t, k=k: t.get(k))
    df["tags"] = df["tags"].map(lambda t: json.dumps(t, sort_keys=True))
    return df.reset_index(drop=True)


def build_pois(points, polys, fcfg: dict[str, Any], crs: int) -> gpd.GeoDataFrame:
    """POIs from nodes and polygon centroids, one boolean column per category."""
    cats = {c: tag_matcher(r) for c, r in fcfg["poi_categories"].items()}
    poly_pts = polys.copy()
    # Centroids in the projected CRS, then back to WGS84 so both halves concat cleanly.
    poly_pts["geometry"] = poly_pts.geometry.to_crs(crs).centroid.to_crs(4326)
    df = pd.concat([points, poly_pts], ignore_index=True)
    for c, m in cats.items():
        df[f"cat_{c}"] = df["tags"].map(m)
    cat_cols = [f"cat_{c}" for c in cats]
    df = gpd.GeoDataFrame(df[df[cat_cols].any(axis=1)], geometry="geometry", crs=4326)
    # Buildings tagged as a POI and a node POI inside them are both kept: they are often
    # distinct (a mall and its shops). Exact duplicates (same OSM object) cannot occur.
    return _finish(df, crs, ["name", "opening_hours", "amenity", "shop"])


def build_transit(points, polys, fcfg: dict[str, Any], crs: int) -> gpd.GeoDataFrame:
    tcfg = fcfg["transit"]
    modes = {m: tag_matcher(tcfg[m]) for m in ("tram_stop", "bus_stop", "rail_station")}
    poly_pts = polys.copy()
    poly_pts["geometry"] = poly_pts.geometry.to_crs(crs).centroid.to_crs(4326)
    df = pd.concat([points, poly_pts], ignore_index=True)
    df["mode"] = None
    for mode, m in modes.items():
        df.loc[df["tags"].map(m) & df["mode"].isna(), "mode"] = mode
    df = gpd.GeoDataFrame(df[df["mode"].notna()], geometry="geometry", crs=4326)
    recent = {s["name"].lower(): s["opened"] for s in tcfg["recent_stations"]}
    names = df["tags"].map(lambda t: (t.get("name") or "").lower())
    # "Town Hall Station" / "Town Hall" both match "town hall"; only rail stations count.
    df["opened"] = [
        next((d for n, d in recent.items() if n in name), None)
        if mode == "rail_station"
        else None
        for name, mode in zip(names, df["mode"])
    ]
    return _finish(df, crs, ["name"])


def extract(site: Site, fcfg: dict[str, Any]) -> dict[str, int]:
    """Run the extract stage for one site and return row counts per layer."""
    bbox, crs, out = site.bbox, site.crs, site.interim
    log.info("reading walkable ways from %s", site.pbf.name)
    ways = read_walkable_ways(site.pbf, bbox, fcfg).to_crs(crs)
    write_parquet(ways, out / "ways.parquet")

    log.info("reading points and polygons")
    points = _read_layer(site.pbf, "points", bbox)
    polys = _read_layer(site.pbf, "multipolygons", bbox)

    pois = build_pois(points, polys, fcfg, crs)
    write_parquet(pois, out / "pois.parquet")

    is_building = polys["tags"].map(lambda t: t.get("building") not in (None, "no"))
    buildings = _finish(
        polys[is_building].copy(), crs, ["building", "height", "building:levels"]
    )
    write_parquet(buildings, out / "buildings.parquet")

    green = _finish(
        _select(polys, fcfg["green_tags"]), crs, ["name", "leisure", "landuse"]
    )
    write_parquet(green, out / "green.parquet")

    ped = _finish(
        _select(polys, ["highway=pedestrian", "area:highway=pedestrian"]), crs, ["name"]
    )
    write_parquet(ped, out / "ped_areas.parquet")

    # GDAL only turns closed ways into polygons for some tags; a station mapped as a closed way
    # (most underground ones, e.g. Melbourne Central) lands in the line layer instead.
    station_lines = _read_layer(
        site.pbf,
        "lines",
        bbox,
        where="railway = 'station' OR other_tags LIKE '%\"public_transport\"=>\"station\"%'",
    )
    transit = build_transit(points, pd.concat([polys, station_lines]), fcfg, crs)
    write_parquet(transit, out / "transit.parquet")

    counts = {
        "ways": len(ways),
        "pois": len(pois),
        "buildings": len(buildings),
        "green": len(green),
        "ped_areas": len(ped),
        "transit": len(transit),
    }
    write_meta(
        out / "extract_meta.json",
        site=site.name,
        pbf=site.pbf.name,
        pbf_sha256=pbf_sha256(site),
        bbox=list(bbox),
        crs=crs,
        config_hash=config_hash(fcfg),
        rows=counts,
    )
    return counts
