import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from src.features import schema
from src.features.common import load_features_config
from src.pipeline import validate_segments as vs

FCFG = load_features_config()
RADII = FCFG["network_metrics"]["radii_m"]


def good_table(n=4):
    """A segment table with every declared column and plausible values."""
    data = {}
    for c in schema.segment_features(FCFG):
        if c.name == "geometry":
            continue
        if c.dtype == "bool":
            data[c.name] = np.ones(n, dtype=bool)
        elif c.dtype == "str":
            data[c.name] = pd.array([f"{c.name}-{i}" for i in range(n)], dtype="str")
        elif c.dtype == "int64":
            data[c.name] = np.ones(n, dtype="int64")
        else:
            data[c.name] = np.full(n, 0.5)
    df = pd.DataFrame(data)
    for r in RADII:  # reach grows with the radius
        df[f"met_reach_{r}"] = float(r)
    geom = [LineString([(i, 0), (i + 1, 0)]) for i in range(n)]
    return gpd.GeoDataFrame(df, geometry=geom, crs=28355)


def failed(checks):
    return {c.name for c in checks if not c.passed}


def test_a_good_table_passes():
    assert failed(vs.validate_segments(good_table(), FCFG)) == set()


def test_duplicate_segments_and_missing_metrics_fail():
    df = good_table()
    df.loc[1, "segment_id"] = df.loc[0, "segment_id"]
    df.loc[2, f"met_betweenness_{RADII[0]}"] = np.nan
    assert {"keys: segment_id unique", "metrics present"} <= failed(
        vs.validate_segments(df, FCFG)
    )


def test_reach_that_shrinks_with_the_radius_fails():
    df = good_table()
    df.loc[0, f"met_reach_{RADII[-1]}"] = 0.0
    assert "reach grows with radius" in failed(vs.check_metrics(df, RADII))


@pytest.mark.parametrize(
    "column,value",
    [("building_coverage", 1.5), ("length_m", 0.0), ("tram_stop_dist_m", 1e9)],
)
def test_values_out_of_range_fail(column, value):
    df = good_table()
    df.loc[0, column] = value
    assert failed(vs.check_ranges(df, FCFG["built_form"]["max_distance_m"])) == {
        "ranges"
    }


def test_a_fragmented_network_warns_but_does_not_fail():
    df = good_table()
    df["in_main_component"] = [True, False, False, False]
    check = vs.check_connectivity(df)
    assert not check.passed and check.severity == vs.WARNING


def test_a_missing_column_fails_the_schema():
    df = good_table().drop(columns="lit")
    assert "schema: columns" in failed(vs.validate_segments(df, FCFG))
