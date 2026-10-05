# NEU Boston campus: site layers (demonstration site)

Everything pulled for the Northeastern Boston campus. Site bbox `sites.neu_boston` in `configs/data.yaml`: campus (~lon -71.095 to -71.082, lat 42.333 to 42.345) plus a ~400 m buffer. Pulled 2026-10-04. No local pedestrian counts exist, so this site is a demonstration, never an accuracy claim.

| Layer | Command | File | License |
|---|---|---|---|
| Terrain, 1 m | `python -m src.ingest.dem --site neu_boston` | `data/raw/dem/neu_boston_usgs_3dep_1m.tif` | Public domain (USGS); credit requested |
| Buildings + heights | `python -m src.ingest.boston_buildings` | `data/raw/boston_buildings/neu_boston_buildings.geojson` | PDDL 1.0 |
| NAIP 2023, 0.3 m | `python -m src.ingest.imagery` | `data/raw/imagery/neu_boston_naip_2023.tif` | Public domain (USDA FSA) |
| MassGIS 2025, 15 cm | `python -m src.ingest.imagery` | `data/raw/imagery/oq25_19TCG*.zip` (8 tiles) | Public domain (MassGIS FAQ); credit "MassGIS (Bureau of Geographic Information), Commonwealth of Massachusetts EOTSS" |
| OSM | `python -m src.ingest.osm --site neu_boston` | `data/raw/osm/neu_boston.osm.pbf` | ODbL 1.0 |
| WorldCover 2021 | `python -m src.ingest.worldcover --site neu_boston` | `data/raw/worldcover/neu_boston_worldcover_2021.tif` | CC BY 4.0 |
| Weather (ERA5) | `python -m src.ingest.weather --site neu_boston` | `data/raw/weather/neu_boston_openmeteo_YYYY.json` | CC BY 4.0; free API non-commercial |

## Terrain (USGS 3DEP 1 m)

- ImageServer clip: 1950 × 2268 px at exactly 1.0 m, EPSG:26919, Float32. Elevation −7.4 to 39.3 m (mean 4.8 m). No nodata pixels.
- The ImageServer is a dynamic mosaic and doesn't report which lidar project fills each pixel. If a citation needs a specific project, the staged tiles are `MA_CentralEastern_2021_B21` `x32y469` and `x32y470` (~410 MB each).
- Bare earth, so it's fine for slope. Negative values are low ground near the Fens/Muddy River, not errors; check before clipping.

## Buildings (Boston Buildings with Roof Breaks)

- 4,239 footprints in the bbox, equal to the server's count, unique `OBJECTID`, no missing geometry.
- Heights `BLDG_HGT_2010`, `ROOF_ELEV_2010`, `GRND_ELEV_2010` are **US feet** (source CRS is EPSG:6492, US ft). Inferred, not documented; convert to metres in features.
- **228 (5.4%) have a missing or zero height.** Fill from OSM `height`/`building:levels`, or from a lidar DSM minus the 3DEP ground.
- Heights come from a **2010 survey**, updated periodically. The newest `Added` date is 2023-06-20, so recent campus buildings may be missing or have stale heights. Spot-check the newest NEU buildings (e.g., EXP) before using them for shade or viewsheds.
- Ingest note: offset paging on this FeatureServer repeats and skips rows, so the script fetches all IDs first, then the features in ID batches.

## Imagery

- **NAIP 2023** (Planetary Computer item `ma_m_4207148_nw_19_030_20230707_20231116`): acquired 2023-07-07, **leaf-on**, 0.3 m, 4 bands (RGBN), EPSG:26919, 6498 × 7558 px clip (160 MB). Read by window from the cloud COG, so the 1.8 GB quarter-quad was never downloaded. Earlier years for this tile: 2021, 2018, 2016 (0.6 m), 2014, 2012 (1 m).
- **MassGIS 2025** (spring leaf-off, acquired 2025-03-18 → 04-23): 8 tiles of 1500 m, each 10000 × 10000 px at 0.15 m, 4 bands uint8, JPEG 2000 (`JP2OpenJPEG` driver), **EPSG:6348** (NAD83(2011) / UTM 19N). Reproject to the NAIP grid, or vice versa, before stacking. Lossy compression.
- **For path segmentation, use the MassGIS leaf-off imagery as the primary input.** Paths under trees are hidden in leaf-on NAIP. NAIP is the portable fallback, since other US sites only have NAIP.

## OSM

BBBike has no Boston extract. The `CambridgeMa` extract (lon −71.30 to −70.82, lat 42.18 to 42.59; snapshot 2026-10-02) covers the campus and its surroundings.

## Land cover

WorldCover clip in the bbox: built-up 78.4%, tree cover 18.3%, grass 2.2%, others <1%.

## Weather

Open-Meteo ERA5 at the campus centre, 2009-01-01 → 2026-09-28, local time (America/New_York), 155,520 hours, no nulls or duplicates.
