import math
import re
import sys

import pytest

from src.ingest import dem
from src.ingest.common import REPO_ROOT, load_config

CFG = load_config()


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Run dem.main() without the network: record each download and manifest entry."""
    seen = {"download": [], "record": []}
    monkeypatch.setattr(dem, "raw_dir", lambda cfg, source: tmp_path)
    monkeypatch.setattr(
        dem,
        "download",
        lambda url, dest, params=None, **kw: seen["download"].append(
            (url, dest, params)
        ),
    )
    monkeypatch.setattr(
        dem,
        "record",
        lambda out, dest, url, params=None, license=None, **kw: seen["record"].append(
            (dest.name, license)
        ),
    )

    def run(site):
        monkeypatch.setattr(sys, "argv", ["dem", "--site", site])
        dem.main()
        return seen

    return run


def test_melbourne_requests_the_ga_5m_clip_of_the_site_bbox(calls):
    seen = calls("melbourne")
    url, dest, params = seen["download"][0]
    lon_min, lat_min, lon_max, lat_max = CFG["sites"]["melbourne"]["bbox"]
    res = CFG["dem"]["ga_5m_res_deg"]
    assert url == CFG["dem"]["ga_5m_wcs"]
    assert dest.name == "melbourne_ga_lidar_5m.tif"
    # The WCS only answers in the DEM's native CRS (GDA94, EPSG:4283).
    assert params["crs"] == "EPSG:4283"
    assert params["bbox"] == f"{lon_min},{lat_min},{lon_max},{lat_max}"
    assert (params["resx"], params["resy"], params["format"]) == (res, res, "GeoTIFF")
    assert seen["record"][0] == ("melbourne_ga_lidar_5m.tif", "CC-BY-4.0")


def test_melbourne_global_tiles_keep_their_names_and_licences(calls):
    seen = calls("melbourne")
    tiles = CFG["dem"]["global_tiles"]["melbourne"]
    names = [dest.name for _, dest, _ in seen["download"][1:]]
    # GDAL's SRTM driver reads the tile position from the file name, so it must be kept.
    assert names == [url.rsplit("/", 1)[-1] for url in tiles.values()]
    licences = dict(seen["record"][1:])
    assert licences["S38E144.hgt.gz"] == "public domain"
    assert "Copernicus" in licences[names[0]]


def test_us_site_gets_a_3dep_clip_with_exact_1m_pixels(calls):
    seen = calls("neu_boston")
    assert all(url != CFG["dem"]["ga_5m_wcs"] for url, _, _ in seen["download"])
    url, dest, params = seen["download"][0]
    assert url == CFG["dem"]["usgs_3dep_imageserver"]
    assert dest.name == "neu_boston_usgs_3dep_1m.tif"
    epsg = CFG["sites"]["neu_boston"]["utm_epsg"]
    assert params["bboxSR"] == params["imageSR"] == epsg

    xmin, ymin, xmax, ymax = map(float, params["bbox"].split(","))
    width, height = map(int, params["size"].split(","))
    res = CFG["sites"]["neu_boston"]["dem"]["res_m"]
    # The bbox is widened to whole pixels, so each pixel is exactly res metres.
    assert math.isclose((xmax - xmin) / width, res)
    assert math.isclose((ymax - ymin) / height, res)
    assert seen["record"][0] == ("neu_boston_usgs_3dep_1m.tif", "public domain (USGS)")


def test_dag_and_sensor_steepness_look_for_the_file_dem_writes():
    # pull_static_layers skips the download when this file is held, and sensor_steepness reads
    # it; if dem.py renamed it, the DAG would re-download daily and steepness would fail.
    dag = (REPO_ROOT / "dags" / "stoa_pipeline.py").read_text()
    steepness = (REPO_ROOT / "src" / "features" / "sensor_steepness.py").read_text()
    assert "dem/melbourne_ga_lidar_5m.tif" in dag
    assert re.search(r"_ga_lidar_5m\.tif", steepness)


def test_a_large_clip_is_requested_in_tiles_that_cover_it():
    parts = dem.tiles(0.0, 0.0, 4500, 3000, 10.0, 2000)
    assert len(parts) == 6  # 3 across, 2 up
    assert max(w for _, _, w, _, _ in parts) <= 2000
    assert sum(w * h for _, _, w, h, _ in parts) == 4500 * 3000
    # The tiles sit on one grid: each starts where the previous one ends.
    assert sorted({x for x, _, _, _, _ in parts}) == [0.0, 20000.0, 40000.0]


def test_neu_stays_one_request_and_nyc_comes_in_tiles():
    assert len(dem.tiles(0.0, 0.0, 1950, 2268, 1.0, 2500)) == 1
    assert len(dem.tiles(0.0, 0.0, 3600, 3500, 10.0, 2500)) == 4


def test_a_site_without_a_dem_source_fails_loudly(calls, monkeypatch):
    sites = {**CFG["sites"], "nowhere": {"bbox": [0, 0, 1, 1]}}
    monkeypatch.setattr(dem, "load_config", lambda: {**CFG, "sites": sites})
    with pytest.raises(SystemExit, match="dem.source"):
        calls("nowhere")
