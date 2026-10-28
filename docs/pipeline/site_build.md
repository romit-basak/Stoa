# Site builds

Melbourne's graph and features are built by the daily `stoa_pipeline`, which also needs its counts. Every other site the app serves, or the transfer test needs, is built by `stoa_site_build` (`dags/stoa_site_build.py`), triggered by hand with `site` set:

| Site | Why | Extra layers |
|---|---|---|
| `neu_boston` | The demonstration (no local counts) | City of Boston building heights, NAIP 2023 and MassGIS 2025 imagery |
| `nyc` | The cross-city transfer test (NYC DOT counts are evaluation only) | None |

Tasks: pull OSM, DEM, WorldCover, weather, calendars and the site's extra layers (`sites.<site>.extra_ingest` in `configs/data.yaml`); run the OSM stages (extract, graph, features, shade; each skipped when nothing it reads changed); check the segment table; version the data. A site is configured in five places, and `tests/test_site_config.py` checks they agree: `sites`, `osm.extracts`, `worldcover.tiles` and `calendars.holidays` in `configs/data.yaml`, and `sites` in `configs/features.yaml`.

## Segment checks (`src/pipeline/validate_segments.py`)

Errors fail the task, so a broken table is never versioned: the columns and types in `src/features/schema.py`; unique `segment_id`; betweenness and reach present for every segment in the metric graph; reach that never shrinks as the radius grows; lengths, coverage, counts and distances in range. A warning: under 80% of segment length in the main connected network. The report is `data/processed/validation/<site>/segments_validation.json`.

## NEU Boston, 25 October 2026

| | |
|---|---|
| Walkable ways in the bbox | 4,330 |
| Segments | 9,345 (with 3 inferred crossings and the plaza edges) |
| Length in the main network | 99.9% |
| Separate-sidewalk roads left out of the metrics | 332 of 409 (77 kept as the only link somewhere) |
| Entrances | 198 (61 linked in the routing graph, 23 edges split) |
| Build time | about 25 s for the OSM stages |

Massachusetts Avenue and Ruggles Street lead angular choice, as expected for the campus edge.

## NYC, 25 October 2026

The transfer test needs features only at NYC DOT's 114 count locations, so the build is clipped (`sites.nyc.feature_mask` in `configs/features.yaml`). The extract keeps everything within 1.9 km of a count location: the 200 m mask radius, plus the largest metric radius (1,600 m), plus 100 m. A segment inside the mask therefore gets the same network metrics it would get from a city-wide graph. Segments in the mask are flagged `in_feature_mask`, and shade is computed only for them. NYC's DEM is USGS 3DEP at 10 m, fetched in tiles of at most 2,500 pixels a side.

| | |
|---|---|
| Walkable ways in the clip | 251,306 |
| Segments | 594,712, of which 33,013 are in the mask |
| Length in the main network | 90.5% (the clip cuts the network into islands at its edge) |
| Left out of the metric graph | 19,690 segments |
| Inferred crossings, plaza edges | 68, 13,146 |
| Entrances | 6,201 (3,984 linked in the routing graph, 2,080 edges split) |
| OSM building heights | 583,669 tagged, 843 from levels, 9,893 at the 9 m default |

| Stage | Time | Peak memory |
|---|---|---|
| Extract | 104 s | 3.1 GB |
| Graph | 102 s | 1.6 GB |
| Features | 12 min (angular choice at 1,600 m takes 7 of them) | 4.8 GB |
| Shade | 5 min | 3.1 GB |

Memory doesn't follow segment count: Melbourne's 111,465 segments peak at 3.3 GB in the features stage, because its dense centre reaches more segments within 1,600 m. So the stage's preflight is a plain limit, `network_metrics.max_segments` (800,000). A site over it stops at once and needs a feature mask or a larger machine.

All segment checks pass. The NYC counts are never read by the training frame (`check_site` rejects every site except the one the hourly table is built for, Melbourne).
