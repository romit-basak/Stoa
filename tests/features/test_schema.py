import re

import pandas as pd
import pytest

from src.features import schema
from src.ingest.common import REPO_ROOT

TABLES = schema.tables()
PROCESSED = REPO_ROOT / "data" / "processed" / "features" / "melbourne"


def test_committed_counts_schema_is_generated_from_schema_py():
    committed = (REPO_ROOT / "configs" / "schemas" / "counts_hourly.yaml").read_text()
    assert committed == schema.schema_yaml("hourly counts", TABLES["counts_hourly"])


@pytest.mark.parametrize("table", list(TABLES))
def test_no_id_coordinate_or_calendar_position_is_a_feature(table):
    # CLAUDE.md core rule 9: no IDs, raw coordinates, month or day of year in a model.
    banned = re.compile(
        r"(^|_)(id|lat|lon|latitude|longitude|month|doy|day_of_year)($|_)"
    )
    feats = schema.features(TABLES[table])
    assert feats
    assert [f for f in feats if banned.search(f)] == []


def test_the_target_and_what_adds_up_to_it_are_never_features():
    roles = {c.name: c.role for c in TABLES["counts_hourly"]}
    assert roles["pedestriancount"] == "label"
    assert roles["direction_1"] == roles["direction_2"] == "aux"


def test_melbourne_only_features_are_marked():
    rich = {
        c.name
        for t in TABLES.values()
        for c in t
        if c.role == "feature" and not c.portable
    }
    assert {"post_metro_tunnel", "slope_pct", "recent_station_dist_m"} <= rich
    portable = schema.features(TABLES["sensor_osm_features"], portable_only=True)
    assert "recent_station_dist_m" not in portable
    assert "met_betweenness_400" in portable


def test_portable_is_set_for_features_only():
    with pytest.raises(ValueError, match="portable"):
        schema.Column("x", "float64", "meta", "osm", portable=True)
    with pytest.raises(ValueError, match="portable"):
        schema.Column("x", "float64", "feature", "osm")


@pytest.mark.parametrize("table", ["osm_segment_features", "sensor_osm_features"])
def test_built_tables_match_the_schema(table):
    path = PROCESSED / f"{table}.parquet"
    if not path.exists():
        pytest.skip("run python -m src.features.osm (or dvc pull) first")
    if table == "osm_segment_features":
        import geopandas as gpd

        df = gpd.read_parquet(path)
    else:
        df = pd.read_parquet(path)
    want = [(c.name, c.dtype) for c in TABLES[table]]
    assert [(c, str(df[c].dtype)) for c in df.columns] == want
