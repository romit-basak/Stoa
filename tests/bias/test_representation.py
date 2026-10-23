import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from src.bias.representation import network_reference, slope_reference
from src.bias.slices import UNKNOWN, load_bias_config

CFG = load_bias_config()

# A patch of Melbourne in GDA94 / MGA 55 (metres), so reprojecting to WGS84 and back is real.
X, Y = 320_000, 5_812_000


def blocks(crs=28355):
    out = gpd.GeoDataFrame(
        {"area_name": ["CBD East", "CBD West", "Carlton"]},
        geometry=[
            box(X, Y, X + 100, Y + 100),
            box(X + 100, Y, X + 200, Y + 100),
            box(X, Y + 100, X + 100, Y + 200),
        ],
        crs=28355,
    )
    return out.to_crs(crs)


def edges():
    return gpd.GeoDataFrame(
        {"length_m": [40.0, 30.0, 25.0, 60.0, 500.0]},
        geometry=[
            LineString([(X + 10, Y + 10), (X + 50, Y + 10)]),  # CBD East
            LineString([(X + 120, Y + 50), (X + 150, Y + 50)]),  # CBD West
            LineString([(X + 10, Y + 150), (X + 35, Y + 150)]),  # Carlton
            # Crosses from CBD East into CBD West: counted once, where its midpoint is.
            LineString([(X + 80, Y + 20), (X + 140, Y + 20)]),
            LineString([(X + 900, Y + 900), (X + 1400, Y + 900)]),  # outside every area
        ],
        crs=28355,
    )


def test_network_reference_sums_length_per_precinct_with_cbd_areas_merged():
    ref = network_reference(edges(), blocks(), CFG)
    assert ref.to_dict() == {"CBD": 40.0 + 30.0 + 60.0, "Carlton": 25.0}


def test_network_reference_counts_each_edge_once_and_drops_edges_outside_clue():
    ref = network_reference(edges(), blocks(), CFG)
    # The 60 m edge counts once; the 500 m one outside every area not at all.
    assert ref.sum() == 40.0 + 30.0 + 25.0 + 60.0


def test_network_reference_reprojects_the_areas_to_the_network_crs():
    in_wgs84 = network_reference(edges(), blocks(crs=4326), CFG)
    assert in_wgs84.to_dict() == network_reference(edges(), blocks(), CFG).to_dict()


def test_slope_reference_sums_surveyed_length_per_band():
    sections = pd.DataFrame(
        {
            "gradepc": [0.5, 0.99, 1.0, 2.9, 3.0, 4.99, 5.0, 12.0, np.nan],
            "distance": [10.0, 5.0, 20.0, 1.0, 7.0, 3.0, 8.0, 2.0, 4.0],
        }
    )
    ref = slope_reference(sections, CFG)
    # Bands are [edge, next edge): 1% is in 1-3%, 5% in >=5%. A missing grade is unknown.
    assert ref.to_dict() == {
        "<1%": 15.0,
        "1-3%": 21.0,
        "3-5%": 10.0,
        ">=5%": 10.0,
        UNKNOWN: 4.0,
    }
