# City of Melbourne open data: context layers

Everything in `data/raw/melbourne_portal/` other than the pedestrian counts. All of it is **CC BY 4.0** unless noted. Attribution: "Source: City of Melbourne Open Data". Pulled 2026-10-03 with `python -m src.ingest.melbourne_portal`. Per-file URL, SHA-256 and portal `modified` date are in `MANIFEST.json`; full portal metadata is in `<dataset>.meta.json`.

Coverage for all layers is the City of Melbourne municipality (~lon 144.897–144.991, lat -37.851 to -37.775).

## Network

| Dataset | Rows | Notes |
|---|---|---|
| `pedestrian_network_full.zip` (alternative export) | 71,060 links + 14,266 property centroids | **Candidate graph.** 2019 vintage. `TYPE`: 1 footpath (27,366), 2 long-wait crossing (789), 3 short-wait crossing (336), 4 zebra (26), 5 tram crossing (32), 6 arcade (580), 7 lane (843), 91 entrance connector, 92 centroid connector. Also `COST`, `MCCID*` (links to footpath assets), `OTIME`/`CTIME` (all defaults 07:00–21:00, not real data). Median link ≈ 10 m; 1,235 km total. **Use this, not `pedestrian-network.geojson`; the records export drops every attribute.** See `pedestrian_network_metadata.pdf`. |
| `footpaths.geojson` | 12,222 polygons | Footpath outlines (2023). Width proxy. |
| `footpath-steepness.geojson` | 33,584 polygons | **Surveyed gradient per footpath section** (`gradepc`, `rlmin`/`rlmax` in AHD). Median 2.1%; 4,455 rows lack a grade; max 580% is an artefact (cap it). Better than any DEM for slope inside CoM. |
| `road-segment.geojson` | 3,974 | Road corridors; joins to parking/street data |
| `tram-tracks.geojson` | 645 | Crossing friction / tram stop proximity |
| `bus-stops.geojson` | 309 | |

## Built form

| Dataset | Rows | Notes |
|---|---|---|
| `2023-building-footprints.geojson` | 41,701 stacked polygons (19,942 structures) | AHD min/max elevation and extrusion per footprint tier; 0 nulls. **Capture dates are mixed** (~79% from 2018-05, rest 2020–2023). Primary input for viewshed and building heights. |
| `buildings-with-name-age-size-…parquet` | 305,557 (census years 2002–2024; 14,094 in 2024) | CLUE: floors, construction year, predominant space use |
| `blocks-for-census-of-land-use-and-employment-clue.geojson` | 603 | CLUE blocks: join key for employment and floor space |
| `employment-by-block-by-clue-industry.parquet` | 13,519 | Jobs per block per year by industry: **trip-generation feature** |

## Destinations / POIs

| Dataset | Rows | Notes |
|---|---|---|
| `business-establishments-with-address-…parquet` | 413,550 (2002–2024) | Point locations with ANZSIC industry; a better POI base than OSM inside CoM. No opening hours. |
| `cafes-and-restaurants-with-seating-capacity.parquet` | 66,356 (2002–2024) | Indoor/outdoor seat counts: hospitality intensity |
| `landmarks-and-places-of-interest-…geojson` | 242 | Theatres, galleries, sports facilities, schools |
| `street-furniture-…geojson` | 3,771 | Seats, fountains, bins |
| `public-toilets.geojson` | 74 | Name, male/female/wheelchair/baby facilities, operator. No opening hours. |
| `drinking-fountains.geojson` | 338 | Fountain assets (type, owner). Weather-interaction feature (hot days). |

## Shade / greenery

| Dataset | Rows | Notes |
|---|---|---|
| `tree-canopies-2021-urban-forest.geojson` | 57,980 polygons (271 MB) | Canopy cover for shade features. Convert to GeoParquet in features. |
| `trees-with-species-and-dimensions-urban-forest.geojson` | 82,064 points | Height, diameter and age class: tree obstruction for viewshed |

## Events

| Dataset | Rows | Notes |
|---|---|---|
| `venues-for-event-bookings.geojson` | 206 | Bookable outdoor venues (parks, squares). No schedule. |
| `event-permits-2014-2018-…parquet` | 2,827 | Event permits with start/end and location text. **Historical only (2014–2018)**, so it overlaps the archive era only. |

## Weather / microclimate

| Dataset | Rows | Notes |
|---|---|---|
| `bom-weather-at-olympic-park.parquet` | 11,359 | 30-min BoM obs, 2020-08 → 2021-04 only. **No license stated.** Validation of the reanalysis only. |
| `microclimate-sensors-data.parquet` | 1,008,189 | 12 CoM street-level sensors, 2022-05 → 2026-09 (temp, humidity, wind, etc.). Useful to check that reanalysis matches street conditions. |

## Not downloaded (large; on request)

| Dataset | Size | Notes |
|---|---|---|
| Digital Surface Model 2018 (0.1 m) | 12 GB zip | `--large`. Tile index in `dsm_tile_index.kml`. Includes buildings/trees (surface, not ground). |
| 3D Point Cloud 2018 (7.5 cm, LAS, unclassified) | multi-GB | `--large`. The only open sub-metre source of bare earth (needs PDAL ground classification). Tile index in `point_cloud_tile_index.kml`. |
| 3D textured mesh (Photomesh) 2018/2020 | — | Visualisation for the 3D app view |
