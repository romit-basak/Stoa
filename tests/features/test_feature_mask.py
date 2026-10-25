import dataclasses

import geopandas as gpd
from shapely.geometry import LineString, Point

from src.features import common
from src.features.common import clip_area, in_feature_mask, site_settings

CRS = 32618


def masked_site(tmp_path, fcfg):
    """NYC's settings with a mask of two points 10 km apart."""
    pts = gpd.GeoDataFrame(
        geometry=[Point(585_000, 4_510_000), Point(595_000, 4_510_000)], crs=CRS
    ).to_crs(4326)
    path = tmp_path / "points.geojson"
    pts.to_file(path)
    cfg = {
        **fcfg,
        "sites": {
            **fcfg["sites"],
            "nyc": {"crs": CRS, "feature_mask": {"points": str(path), "radius_m": 200}},
        },
    }
    return site_settings("nyc", feat_cfg=cfg), cfg


def test_the_clip_reaches_the_mask_plus_the_largest_metric_radius(tmp_path, fcfg):
    site, cfg = masked_site(tmp_path, fcfg)
    area = clip_area(site, cfg)
    margin = 200 + max(cfg["network_metrics"]["radii_m"]) + 100
    assert area.contains(Point(585_000 + margin - 10, 4_510_000))
    assert not area.contains(Point(585_000 + margin + 10, 4_510_000))
    assert not area.contains(
        Point(590_000, 4_510_000)
    )  # between the two, far from both


def test_segments_near_a_mask_point_are_in_the_mask(tmp_path, fcfg):
    site, cfg = masked_site(tmp_path, fcfg)
    edges = gpd.GeoDataFrame(
        geometry=[
            LineString([(585_050, 4_510_000), (585_100, 4_510_000)]),
            LineString([(585_500, 4_510_000), (585_600, 4_510_000)]),
        ],
        crs=CRS,
    )
    assert in_feature_mask(edges, site, cfg).tolist() == [True, False]


def test_without_a_mask_every_segment_counts(fcfg):
    site = dataclasses.replace(site_settings("melbourne"))
    edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=28355)
    assert in_feature_mask(edges, site, fcfg).tolist() == [True]
    assert common.clip_area(site, fcfg) is None
