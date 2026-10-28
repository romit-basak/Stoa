"""A site is configured in five places; a site missing from any of them breaks its build."""

import pytest

from src.features.common import load_features_config
from src.ingest.common import load_config

DATA = load_config()
FEATURES = load_features_config()
SITES = list(DATA["sites"])


@pytest.mark.parametrize("site", SITES)
def test_every_site_is_configured_everywhere(site):
    s = DATA["sites"][site]
    assert {"bbox", "center", "timezone", "dem"} <= set(s), site
    lon_min, lat_min, lon_max, lat_max = s["bbox"]
    assert lon_min < s["center"][0] < lon_max and lat_min < s["center"][1] < lat_max
    assert site in DATA["osm"]["extracts"]
    assert site in DATA["worldcover"]["tiles"]
    assert site in DATA["calendars"]["holidays"]
    assert "crs" in FEATURES["sites"][site]
    if s["dem"]["source"] == "usgs_3dep":
        assert "utm_epsg" in s and "res_m" in s["dem"]


def test_no_site_is_configured_only_in_some_places():
    for place in (
        DATA["osm"]["extracts"],
        DATA["worldcover"]["tiles"],
        DATA["calendars"]["holidays"],
        FEATURES["sites"],
    ):
        assert set(place) <= set(SITES)


def test_only_the_training_site_counts_sensors():
    counting = [s for s in SITES if DATA["sites"][s].get("count_sensors")]
    assert counting == [FEATURES["counts_hourly"]["site"]]
