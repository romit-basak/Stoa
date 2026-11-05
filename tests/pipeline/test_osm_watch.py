import geopandas as gpd
from shapely.geometry import LineString, Point

from src.pipeline import osm_watch

TAGS = ["highway", "surface"]


def ways(*rows):
    return gpd.GeoDataFrame(
        [
            {"way_id": w, "node_ids": nodes, "highway": hw, "surface": s, "geometry": g}
            for w, nodes, hw, s, g in rows
        ],
        geometry="geometry",
        crs=4326,
    )


LINE = LineString([(144.960, -37.810), (144.961, -37.810)])
FAR = LineString([(144.990, -37.830), (144.991, -37.830)])


def test_changes_are_added_removed_rerouted_or_retagged():
    old = ways(
        (1, [1, 2], "footway", "asphalt", LINE),
        (2, [3, 4], "footway", None, LINE),
        (3, [5, 6], "path", "gravel", LINE),
        (4, [7, 8], "footway", None, LINE),
    )
    new = ways(
        (1, [1, 2], "footway", "asphalt", LINE),  # unchanged
        (2, [3, 9, 4], "footway", None, LINE),  # a node added: re-routed
        (3, [5, 6], "path", "asphalt", LINE),  # surface changed
        (5, [10, 11], "footway", None, LINE),  # new
    )
    c = osm_watch.way_changes(old, new, TAGS).set_index("way_id")
    assert c["change"].to_dict() == {
        2: "rerouted",
        3: "retagged",
        4: "removed",
        5: "added",
    }
    assert c.loc[3, "tags"] == "surface"


def test_missing_tags_on_both_sides_are_not_a_change():
    old = ways((1, [1, 2], "footway", None, LINE))
    new = ways((1, [1, 2], "footway", None, LINE))
    assert osm_watch.way_changes(old, new, TAGS).empty


def test_only_changes_near_a_sensor_are_reported():
    changes = gpd.GeoDataFrame(
        {"way_id": [1, 2], "change": ["added", "removed"], "tags": [None, None]},
        geometry=[LINE, FAR],
        crs=4326,
    )
    sensors = gpd.GeoDataFrame(
        {"location_id": [7]}, geometry=[Point(144.9605, -37.8102)], crs=4326
    )
    near = osm_watch.near_sensors(changes, sensors, crs=28355, radius_m=100)
    assert near["way_id"].tolist() == [1]
    assert near["distance_m"].iloc[0] < 100


def test_every_change_near_a_sensor_is_listed_not_only_the_nearest():
    close = LineString([(144.9604, -37.8101), (144.9606, -37.8101)])
    changes = gpd.GeoDataFrame(
        {"way_id": [1, 2], "change": ["added", "retagged"], "tags": [None, "lit"]},
        geometry=[LINE, close],
        crs=4326,
    )
    sensors = gpd.GeoDataFrame(
        {"location_id": [7]}, geometry=[Point(144.9605, -37.8102)], crs=4326
    )
    near = osm_watch.near_sensors(changes, sensors, crs=28355, radius_m=100)
    assert sorted(near["way_id"]) == [1, 2]


def test_the_latest_extract_never_replaces_the_pinned_one(monkeypatch, tmp_path):
    # The watch downloads into a temporary folder, not data/raw/osm.
    seen = []

    def fake_download(url, dest, **kw):
        seen.append(dest)
        raise RuntimeError("stop after the download call")

    monkeypatch.setattr(osm_watch, "download", fake_download)
    monkeypatch.setattr("sys.argv", ["osm_watch"])
    try:
        osm_watch.main()
    except RuntimeError:
        pass
    assert seen and "data/raw/osm" not in str(seen[0])
