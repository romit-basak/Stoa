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
