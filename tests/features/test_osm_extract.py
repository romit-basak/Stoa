import pytest

from src.features.osm_extract import (
    _read_layer,
    build_pois,
    build_transit,
    is_walkable,
    parse_hstore,
    read_walkable_ways,
)
from tests.features.conftest import BBOX, CRS


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"highway": "footway"}, True),
        ({"highway": "residential"}, True),
        ({"highway": "motorway"}, False),
        ({"highway": "footway", "area": "yes"}, False),
        ({"highway": "service", "access": "private"}, False),
        ({"highway": "service", "service": "private"}, False),
        ({"highway": "footway", "foot": "no"}, False),
        # An explicit foot permission beats access=no.
        ({"highway": "footway", "access": "no", "foot": "yes"}, True),
        ({"highway": "cycleway"}, False),
        ({"highway": "cycleway", "foot": "designated"}, True),
        ({"highway": "construction"}, False),
        ({"building": "yes"}, False),
    ],
)
def test_walk_filter(tags, expected, fcfg):
    assert is_walkable(tags, fcfg["walk_filter"]) is expected


def test_parse_hstore_handles_escaped_quotes():
    s = '"amenity"=>"cafe","name"=>"The \\"Best\\" Cafe"'
    assert parse_hstore(s) == {"amenity": "cafe", "name": 'The "Best" Cafe'}
    assert parse_hstore(None) == {}
    assert parse_hstore(float("nan")) == {}


def test_read_walkable_ways(osm_file, fcfg):
    ways = read_walkable_ways(osm_file, BBOX, fcfg)
    assert sorted(ways.way_id) == [100, 101, 103, 105]
    w100 = ways.set_index("way_id").loc[100]
    assert w100.node_ids == [1, 2, 3]
    assert w100.footway == "sidewalk"


def test_ways_outside_bbox_are_dropped(osm_file, fcfg):
    far_bbox = (150.0, -30.0, 150.1, -29.9)
    assert read_walkable_ways(osm_file, far_bbox, fcfg).empty


def test_pois_get_categories_and_hours(osm_file, fcfg):
    points = _read_layer(osm_file, "points", BBOX)
    polys = points.iloc[0:0]
    pois = build_pois(points, polys, fcfg, CRS).set_index("osm_id")
    assert pois.loc[11, "cat_retail"] and not pois.loc[11, "cat_food_drink"]
    assert pois.loc[12, "cat_food_drink"]
    assert pois.loc[11, "opening_hours"] == "Mo-Fr 09:00-17:00"
    assert 13 not in pois.index  # the tram stop is transit, not a POI
    assert pois.crs.to_epsg() == CRS


def test_transit_modes_and_recent_stations(osm_file, fcfg):
    points = _read_layer(osm_file, "points", BBOX)
    transit = build_transit(points, points.iloc[0:0], fcfg, CRS)
    assert transit.set_index("osm_id").loc[13, "mode"] == "tram_stop"
    assert transit["opened"].isna().all()
