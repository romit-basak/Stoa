import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from src.features.sensor_steepness import (
    DEM,
    FOOTPATH,
    dem_terrain,
    fill_from_dem,
    match_footpaths,
    outdoor_sensors,
    usable_sections,
)

CRS = 28355
X0, Y0 = 320_000.0, 5_810_000.0  # central Melbourne in MGA zone 55


def sensors_at(*xy, ids=None):
    ids = ids or list(range(1, len(xy) + 1))
    return gpd.GeoDataFrame(
        {"location_id": ids, "sensor_description": [f"s{i}" for i in ids]},
        geometry=[Point(X0 + x, Y0 + y) for x, y in xy],
        crs=CRS,
    )


def section(mcc_id, x0, x1, gradepc, length=None, rl=(10.0, 11.0)):
    """A 2 m wide footpath polygon from x0 to x1 along y = 0."""
    return {
        "mcc_id": mcc_id,
        "gradepc": gradepc,
        "rlmin": rl[0] if gradepc is not None else None,
        "rlmax": rl[1] if gradepc is not None else None,
        "distance": length if length is not None else x1 - x0,
        "geometry": box(X0 + x0, Y0 - 1, X0 + x1, Y0 + 1),
    }


def sections_of(*rows):
    return gpd.GeoDataFrame(list(rows), crs=CRS)


def usable(*rows):
    return usable_sections(sections_of(*rows), CRS, min_section_m=5)[0]


def test_sections_without_grade_or_too_short_are_skipped():
    raw = sections_of(
        section(1, 0, 40, None),  # no grade
        section(2, 40, 43, 90.0),  # 3 m long: grades like this are artefacts
        section(3, 43, 80, 2.0),
    )
    kept, stats = usable_sections(raw, CRS, min_section_m=5)
    assert kept["mcc_id"].tolist() == [3]
    assert stats == {
        "sections": 3,
        "sections_no_grade": 1,
        "sections_short": 1,
        "sections_used": 1,
    }
    assert kept["elevation_m"].iloc[0] == pytest.approx(10.5)


def test_sensor_on_an_ungraded_section_takes_the_nearest_measured_one():
    # The notebook matched sensor 143 to an empty section and left it without a slope.
    raw = sections_of(section(1, 0, 40, None), section(2, 45, 80, 3.0))
    kept, _ = usable_sections(raw, CRS, min_section_m=5)
    out = match_footpaths(sensors_at((20, 0)), kept, max_match_m=30, local_radius_m=25)
    row = out.iloc[0]
    assert row["slope_pct"] == 3.0
    assert row["section_id"] == 2
    assert row["match_distance_m"] == pytest.approx(25.0)
    assert row["place_source"] == FOOTPATH


def test_local_mean_is_weighted_by_section_length():
    kept = usable(section(1, -30, 0, 1.0), section(2, 0, 10, 5.0))
    out = match_footpaths(sensors_at((0, 0)), kept, max_match_m=25, local_radius_m=25)
    assert out["slope_pct_local_mean"].iloc[0] == pytest.approx((1 * 30 + 5 * 10) / 40)
    assert out["n_local_sections"].iloc[0] == 2


def test_sensor_beyond_the_match_distance_is_left_for_the_dem():
    kept = usable(section(1, 0, 40, 2.0))
    out = match_footpaths(
        sensors_at((20, 100)), kept, max_match_m=25, local_radius_m=25
    )
    assert out["slope_pct"].isna().all()
    assert out["place_source"].isna().all()
    assert out["n_local_sections"].iloc[0] == 0


@pytest.fixture
def tilted_dem(tmp_path):
    """A 5 m DEM rising 4 m per 100 m eastward and 3 m per 100 m northward (5% slope)."""
    res, n = 5.0, 80
    left, top = X0 - 200, Y0 + 200
    cx = left + res * (np.arange(n) + 0.5)
    cy = top - res * (np.arange(n) + 0.5)
    z = 10 + 0.04 * (cx[None, :] - X0) + 0.03 * (cy[:, None] - Y0)
    path = tmp_path / "dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs=f"EPSG:{CRS}",
        transform=from_origin(left, top, res, res),
    ) as dst:
        dst.write(z.astype("float32"), 1)
    return path


def test_dem_terrain_recovers_a_known_plane(tilted_dem):
    pts = sensors_at((2.5, 2.5)).geometry
    elevation, slope = dem_terrain(pts, tilted_dem, offset_m=10)
    assert elevation[0] == pytest.approx(10 + 0.04 * 2.5 + 0.03 * 2.5, abs=0.01)
    assert slope[0] == pytest.approx(5.0, abs=0.01)


def test_unmatched_sensors_are_filled_from_the_dem(tilted_dem):
    sensors = sensors_at((2.5, 2.5), (50, 50), ids=[25, 7])
    table = pd.DataFrame(
        {
            "location_id": [25, 7],
            "slope_pct": [np.nan, 1.5],
            "elevation_m": [np.nan, 9.0],
            "place_source": [None, FOOTPATH],
        }
    )
    out = fill_from_dem(table, sensors, tilted_dem, offset_m=10)
    assert out["place_source"].tolist() == [DEM, FOOTPATH]
    assert out["slope_pct"].iloc[0] == pytest.approx(5.0, abs=0.01)
    assert out.iloc[1]["slope_pct"] == 1.5  # matched sensors keep the footpath value


def test_only_outdoor_sensors_are_kept(tmp_path):
    path = tmp_path / "sensors.geojson"
    gpd.GeoDataFrame(
        {
            "location_id": [1, 2, 2],
            "sensor_description": ["a", "b", "b"],
            "location_type": ["Outdoor", "Indoor", "Indoor"],
        },
        geometry=[Point(144.96, -37.81)] * 3,
        crs=4326,
    ).to_file(path)
    out = outdoor_sensors(path, CRS)
    assert out["location_id"].tolist() == [1]
    assert out.crs.to_epsg() == CRS
