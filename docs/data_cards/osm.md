# OpenStreetMap (Melbourne extract)

| | |
|---|---|
| **Role** | Walking network (graph), POIs + opening hours, crossings, entrances, building outlines; Boston campus network later |
| **Provider** | OpenStreetMap contributors, via BBBike weekly city extract |
| **License** | ODbL 1.0. Attribution: "© OpenStreetMap contributors". Share-alike applies if we *distribute* a derived database (e.g. a published graph). Model outputs (produced works) need attribution only. |
| **Access** | `python -m src.ingest.osm --site melbourne` → `data/raw/osm/melbourne.osm.pbf` (89 MB) |
| **Snapshot** | Extract `Last-Modified` 2026-09-26 (see `MANIFEST.json`) |
| **Other sites** | `--site neu_boston` → BBBike `CambridgeMa` (no Boston extract exists; covers lon −71.30 to −70.82, lat 42.18 to 42.59). `--site nyc` → BBBike `NewYork` (153 MB; covers all 114 NYC DOT count points). Both snapshot 2026-10-02. |
| **Coverage** | lon 144.68–145.30, lat -38.02 to -37.53 (Melbourne metro, much wider than the study bbox). Clip to `sites.melbourne.bbox` in `src/features`. |

## Contents (counts from 2026-09-26 extract)

Way counts are for the whole metro extract; node counts are within the study bbox.

| Feature | Count |
|---|---|
| `highway=footway` ways | 176,154 |
| `highway=path` / `pedestrian` / `steps` / `living_street` | 10,678 / 1,138 / 3,413 / 363 |
| `highway=corridor` (indoor/arcade links) | 1,163 |
| Highway ways with `surface` / `lit` | 213,217 / 41,552 |
| Buildings (with `height` / `building:levels`) | 483,382 (18,709 / 53,192) |
| Nodes in study bbox | 850,317 |
| …with `crossing` / `entrance` | 10,245 / 1,146 |
| …with `amenity` / `shop` / `opening_hours` | 12,155 / 2,803 / 1,813 |

## Known issues

- Sidewalks are mapped inconsistently: some streets have separate `footway=sidewalk` ways, others only a centre-line road. The graph builder must handle both, or use the CoM Pedestrian Network instead (see `melbourne_portal.md`).
- Building heights are sparse (~4% of buildings). Use CoM 2023 footprints for heights.
- `opening_hours` coverage is low (~2.5k tagged features in the metro area). Treat it as a partial feature with explicit missingness.
- OSM changes continuously. Each training run must pin the extract by its SHA-256 (from `MANIFEST.json` / DVC).
- An earlier Overpass full-bbox dump timed out (HTTP 504), which is why we use BBBike.
