import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from src.features import surroundings as su

CRS = 28355
X0, Y0 = 320_000.0, 5_810_000.0


def segments(*lines):
    return gpd.GeoDataFrame(
        {"segment_id": [f"s{i}" for i in range(len(lines))]},
        geometry=[LineString([(X0 + x, Y0 + y) for x, y in c]) for c in lines],
        crs=CRS,
    )


def raster(path, data, res=1.0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=f"EPSG:{CRS}",
        transform=from_origin(X0, Y0 + data.shape[0] * res, res, res),
    ) as dst:
        dst.write(data, 1)
    return path


def test_slope_is_rise_over_run_along_the_segment(tmp_path):
    # A plane rising 5 m per 100 m eastward.
    x = np.arange(200, dtype="float32")
    dem = raster(tmp_path / "dem.tif", np.tile(0.05 * x, (200, 1)))
    out = su.segment_slope(
        segments([(10, 100), (110, 100)], [(50, 10), (50, 110)]), dem, 10.0
    )
    assert out.loc[0, "dem_slope_mean_pct"] == pytest.approx(
        5.0, abs=0.2
    )  # east: uphill
    assert out.loc[1, "dem_slope_max_pct"] == pytest.approx(
        0.0, abs=0.01
    )  # north: level


def test_landcover_shares_count_the_class_around_the_segment(tmp_path):
    lc = np.full((200, 200), 50, dtype="uint8")  # built everywhere
    lc[:, 100:] = 10  # trees on the east half
    path = raster(tmp_path / "wc.tif", lc)
    out = su.landcover_shares(
        segments([(20, 100), (40, 100)], [(160, 100), (180, 100)]), path, 10
    )
    assert out.loc[0, "wc_built_share"] == pytest.approx(1.0)
    assert out.loc[1, "wc_tree_share"] == pytest.approx(1.0)
    assert out.loc[0, "wc_tree_share"] == pytest.approx(0.0)


def test_missing_inputs_leave_empty_columns(tmp_path):
    edges = segments([(0, 0), (10, 0)])
    assert (
        su.landcover_shares(edges, tmp_path / "none.tif", 50)["wc_tree_share"]
        .isna()
        .all()
    )
    assert su.segment_slope(edges, None, 10)["dem_slope_mean_pct"].isna().all()
    assert np.isnan(su.polygon_cover(edges, None, 25)).all()


def test_polygon_cover_is_the_share_of_the_buffer_covered():
    edges = segments([(0, 0), (100, 0)])
    # A canopy covering the north half of the segment's buffer.
    canopy = gpd.GeoDataFrame(geometry=[box(X0 - 50, Y0, X0 + 150, Y0 + 50)], crs=CRS)
    assert su.polygon_cover(edges, canopy, 10)[0] == pytest.approx(0.5, abs=0.02)


def test_a_short_segment_over_a_step_in_the_dem_is_not_a_cliff(tmp_path):
    dem = np.zeros((50, 50), dtype="float32")
    dem[:, 25:] = 1.0  # a 1 m step, like a kerb or a wall
    path = raster(tmp_path / "step.tif", dem)
    out = su.segment_slope(segments([(24, 25), (26, 25)]), path, 10.0)
    assert out.loc[0, "dem_slope_max_pct"] <= 10.0  # 1 m over at least 10 m


def test_bridges_get_no_ground_slope(tmp_path):
    x = np.arange(100, dtype="float32")
    path = raster(tmp_path / "dem.tif", np.tile(0.1 * x, (100, 1)))
    edges = segments([(10, 50), (60, 50)], [(10, 60), (60, 60)])
    edges["bridge"] = ["yes", None]
    out = su.segment_slope(edges, path, 10.0)
    assert (
        np.isnan(out.loc[0, "dem_slope_mean_pct"])
        and out.loc[1, "dem_slope_mean_pct"] > 5
    )


def test_ground_outside_the_dem_survey_is_missing_not_a_cliff(tmp_path):
    dem = np.full((100, 100), 40.0, dtype="float32")
    dem[:, 50:] = 0.0  # the fill outside the survey, with no nodata declared
    path = raster(tmp_path / "filled.tif", dem)
    edges = segments([(10, 50), (90, 50)], [(60, 20), (90, 20)])
    out = su.segment_slope(edges, path, 10.0, nodata=0)
    assert out.loc[0, "dem_slope_max_pct"] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(out.loc[1, "dem_slope_max_pct"])
