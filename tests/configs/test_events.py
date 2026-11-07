import warnings
from datetime import datetime

import pytest
import yaml

from src.ingest.common import REPO_ROOT

LOG = yaml.safe_load((REPO_ROOT / "configs" / "events.yaml").read_text())
PBF = REPO_ROOT / "data" / "raw" / "osm" / "melbourne.osm.pbf"
KINDS = {
    "cricket",
    "football",
    "basketball",
    "netball",
    "concert",
    "graduation",
    "halloween",
}


def test_every_source_has_a_page_and_an_archived_copy():
    for key, s in LOG["sources"].items():
        assert s["url"].startswith("https://"), key
        assert s["archived"].startswith("https://web.archive.org/web/"), key
        assert s["url"] in s["archived"], key


def test_every_event_is_sourced_placed_and_timed():
    for e in LOG["events"]:
        assert e["source"] in LOG["sources"], e["name"]
        assert e["kind"] in KINDS, e["name"]
        if e.get("scope") == "citywide":
            assert e["venue"] is None, e["name"]
        else:
            assert e["venue"] in LOG["venues"], e["name"]
        start = datetime.fromisoformat(e["start"])
        assert start.tzinfo is None  # Melbourne local clock time
        if e["end"] is not None:
            assert datetime.fromisoformat(e["end"]) > start, e["name"]
        if start.hour == 0 and start.minute == 0 and e["end"] is None:
            # A midnight start with no end means the source gave only a date: say so.
            assert "date only" in e.get("time_note", ""), e["name"]
        assert e["expected_crowd"] is None or e["expected_crowd"] > 0, e["name"]


def test_no_event_is_listed_twice():
    keys = [(e["name"], e["venue"], e["start"]) for e in LOG["events"]]
    assert len(keys) == len(set(keys))


def test_venues_name_an_osm_object():
    for key, v in LOG["venues"].items():
        assert ("osm" in v) != ("osm_name" in v), key
        if "osm" in v:
            kind, _, num = v["osm"].partition("/")
            assert kind in {"way", "node", "relation"} and num.isdigit(), key


def test_venue_ways_exist_in_the_pinned_extract():
    if not PBF.exists():
        pytest.skip("run python -m src.ingest.osm (or dvc pull) first")
    import pyogrio

    ways = [
        int(v["osm"].split("/")[1])
        for v in LOG["venues"].values()
        if v.get("osm", "").startswith("way/")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = pyogrio.read_dataframe(
            PBF,
            layer="multipolygons",
            where="osm_way_id IN ({})".format(",".join(f"'{w}'" for w in ways)),
            columns=["osm_way_id"],
        )
    assert sorted(found["osm_way_id"].astype(int)) == sorted(ways)
