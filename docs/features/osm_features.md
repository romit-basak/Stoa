# OSM features

This pipeline turns the OpenStreetMap extract into a walking graph and one row of features per path segment. It also matches the City of Melbourne count sensors to those segments, so the features join straight onto the counts.

It is the input for the **portable** model (features any city has; see `CLAUDE.md`, "Generalization"). The same code runs for `neu_boston` and `nyc`. Only Melbourne has been run and checked so far.

## Run it

```bash
.venv/bin/python -m src.ingest.osm --site melbourne        # once: download the .osm.pbf
.venv/bin/python -m src.features.osm --site melbourne      # all stages, ~2.5 min on a laptop
.venv/bin/python -m src.features.osm --only features,sensors   # re-run some stages
```

| Stage | Module | Reads | Writes |
|---|---|---|---|
| `extract` | `src/features/osm_extract.py` | `data/raw/osm/<site>.osm.pbf` | `data/interim/osm/<site>/{ways,pois,buildings,green,ped_areas,transit}.parquet` |
| `graph` | `src/features/graph.py` | `ways.parquet` | `data/interim/osm/<site>/graph/{edges,nodes}.parquet` |
| `features` | `src/features/segment_features.py` (+ `network_metrics.py`, `opening_hours.py`) | interim layers and graph | `data/processed/features/<site>/osm_segment_features.parquet`, `osm_segment_open_hours.parquet` |
| `sensors` (Melbourne only) | `src/features/sensor_match.py` | graph, features, CoM sensor locations | `sensor_segment_map.parquet`, `sensor_osm_features.parquet`, `sensor_match_review.csv` |

All settings are in `configs/features.yaml`: walk filter, POI categories, radii, sensor-matching thresholds and the Metro Tunnel station dates. Every spatial file is GeoParquet in the site's metric CRS (EPSG:28355 for Melbourne). Read it with `geopandas.read_parquet`; `pandas.read_parquet` also works if you don't need the geometry. Each stage writes a `*_meta.json` next to its outputs, holding the extract's SHA-256, a hash of the config, row counts and runtimes.

## Joining with the counts

```python
import pandas as pd

counts = pd.read_parquet("data/raw/melbourne_portal/pedestrian-counting-system-monthly-counts-per-hour.parquet")
osm = pd.read_parquet("data/processed/features/melbourne/sensor_osm_features.parquet")
df = counts.merge(osm, on="location_id", how="left", validate="many_to_one")

# Opening hours vary by hour of the week: join them through the matched segment.
# hourday in the counts is local time; Monday = 0.
df["hour_of_week"] = pd.to_datetime(df.sensing_date).dt.dayofweek * 24 + df.hourday
hours = pd.read_parquet("data/processed/features/melbourne/osm_segment_open_hours.parquet")
df = df.merge(hours, on=["segment_id", "hour_of_week"], how="left").fillna(
    {c: 0 for c in hours.columns if c.startswith(("open_", "unknown_hours_"))}
)
```

- The 2009–2022 archive uses `Sensor_ID`, which is the same ID as `location_id`.
- Use outdoor sensors only (`location_type == "Outdoor"`). The 34 indoor sensors are matched for completeness, but OSM's indoor corridors are patchy.
- Filter or check rows where `needs_review` is true until the sensor matches have been hand-checked (see below).
- For predictions on segments with no sensor, use `osm_segment_features.parquet` directly. It has the same columns, keyed on `segment_id`.

## Melbourne run (extract of 2026-09-26)

| | |
|---|---|
| Walkable ways in the study bbox | 47,661 |
| Segments / junctions | 109,733 / 75,379 (median segment 14 m) |
| Length in the largest connected network | 99.4% |
| POIs / dropped (> 50 m from the graph) | 13,029 / 365 |
| POIs with `opening_hours` / parsed | 2,032 / 1,997 (98%) |
| Transit points | 585 tram stops, 788 bus stops, 36 rail stations (incl. the 5 Metro Tunnel stations) |
| Sensors matched | 134 of 134; 109 within 5 m, 131 within 15 m (median 2.1 m) |
| Sensors flagged for review | 79 (34 of them only because they are indoor) |

Rank correlation with each outdoor sensor's mean hourly count (live window, 100 sensors) shows the features carry signal with the expected signs. This is a check that the join works, not a model result:

| Feature | Spearman |
|---|---|
| `poi_food_drink_200`, `poi_retail_400` | +0.59 |
| `met_reach_400` | +0.55 |
| `ang_choice_400` | +0.53 |
| `met_betweenness_400` | +0.49 |
| `tram_stop_dist_m` / `rail_station_dist_m` | −0.39 / −0.40 |

## Column dictionary: `osm_segment_features.parquet`

One row per segment. `segment_id` is `<OSM way id>-<k>`, where `k` numbers the pieces of that way from its first node. IDs are stable for one extract but change when the extract changes, so models and overrides must record the extract's SHA-256. Network distances are measured from the segment's nearer end.

**Identity and geometry**

| Column | Meaning |
|---|---|
| `segment_id`, `way_id` | Segment ID and the OSM way it came from |
| `name`, `highway` | OSM street name (65% missing: most footways have none) and raw highway tag |
| `length_m`, `bearing_deg` | Length; direction from north folded to 0–180 |
| `in_main_component` | On the main connected network (false for islands such as isolated indoor corridors) |
| `geometry` | LineString, site CRS |

**Path type and tags.** Booleans, except where noted.

| Column | Meaning |
|---|---|
| `hw_<class>` | One-hot highway class: footway, path, pedestrian, steps, corridor, living_street, residential, service, unclassified, tertiary, secondary, primary, trunk, cycleway (only those open to pedestrians), track, other. `_link` roads count as their main class. |
| `is_sidewalk`, `is_crossing`, `is_steps`, `is_indoor`, `is_bridge`, `is_tunnel`, `is_covered` | From `footway`, `highway`, `indoor`, `bridge`, `tunnel` and `covered` tags |
| `sidewalk_both`, `sidewalk_one_side`, `sidewalk_none`, `sidewalk_separate`, `sidewalk_missing` | Sidewalk tagging on roads. `separate` means the footpaths are their own segments. |
| `surface_paved`, `lit` | 1/0, NaN when untagged; paired with `surface_paved_missing` and `lit_missing` |
| `width_m`, `width_missing` | Tagged width (97% missing) |

**Amenities within walking distance.** `poi_<category>_<r>` is the number of POIs of that category within `r` = 100, 200 or 400 m along the network. The categories are food_drink, retail, toilets, drinking_water, bench, education, culture_entertainment, healthcare, services, worship and tourism_lodging; the tag rules for each are in `configs/features.yaml`. A POI can count in several categories. Polygon POIs (e.g., a mall) count once, at their centroid.

**Built form and surroundings**

| Column | Meaning |
|---|---|
| `building_coverage`, `building_count` | Share of a 25 m buffer covered by OSM building footprints, and how many footprints touch it |
| `green_dist_m`, `green_adjacent` | Straight-line distance to the nearest park/green polygon (capped at 2,000 m); within 20 m |
| `ped_area_dist_m` | Straight-line distance to the nearest pedestrian plaza polygon (capped at 2,000 m) |

**Transit.** Network distance to the nearest stop, capped at 2,000 m:

| Column | Meaning |
|---|---|
| `tram_stop_dist_m`, `bus_stop_dist_m` | Nearest tram or bus stop |
| `rail_station_dist_m` | Nearest rail station, not counting the recent stations below |
| `recent_station_dist_m` | Nearest Metro Tunnel station (Arden, Parkville, State Library, Town Hall, Anzac; opened 2025-11-30). **Set it to the cap for any hour before the opening date**, or the 2026 snapshot leaks the stations into 2009–2025 counts. |

**Network structure.** Computed for `r` = 400, 800 and 1,600 m.

| Column | Meaning |
|---|---|
| `degree_min`, `degree_max` | Junction degree at the segment's two ends |
| `met_betweenness_<r>` | Shortest-path betweenness: node pairs at most `r` apart whose shortest path runs along the segment (ties split evenly) |
| `met_betweenness_lw_<r>` | The same, but each pair is weighted by the network length around its two ends (km²), so densely mapped areas don't count extra |
| `met_closeness_<r>`, `met_harmonic_<r>` | Closeness (nodes reached ÷ total distance) and harmonic closeness (sum of 1/distance), averaged over the two ends. Closeness is NaN when fewer than 10 nodes are in range. |
| `met_reach_<r>`, `met_reach_len_<r>` | Nodes, and metres of network, within `r` |
| `junctions_<r>` | Junctions (degree ≥ 3) within `r` |
| `ang_choice_<r>`, `ang_choice_lw_<r>` | Angular choice (Space Syntax): betweenness on least-angle routes, plain and length-weighted |
| `ang_node_count_<r>`, `ang_reach_len_<r>`, `ang_total_depth_<r>` | Segments and metres in range; sum of angular depth to them (a 90° turn = 1) |
| `ang_integration_<r>` | Normalised angular integration NAIN = NC^1.2 / TD. NaN when fewer than 10 segments are in range. |
| `ang_choice_norm_<r>` | Normalised angular choice NACH = log(choice + 1) / log(TD + 3) |

In the angular measures the radius is metric: the distance along the least-angle route, measured between segment midpoints. A route can't turn back inside a segment. Betweenness and choice count each pair once. The implementation is checked against networkx and a brute-force enumeration in `tests/features/test_network_metrics.py`.

## `osm_segment_open_hours.parquet`

Key `(segment_id, hour_of_week)`, with `hour_of_week` = weekday × 24 + hour, Monday = 0, local time. For each of food_drink, retail, culture_entertainment and services:

- `open_<category>`: POIs within 200 m that are open at half past that hour
- `unknown_hours_<category>`: POIs within 200 m with no parseable `opening_hours` (the same in every hour)

Only segments with at least one such POI in range appear; a missing row means zero. Opening hours are sparse: about 15% of POIs carry them, so the `unknown_hours_` columns matter. A model should see both. Public-holiday rules are ignored. The parser covers the common OSM forms; anything else (free text, months, sunrise/sunset) is treated as unknown rather than guessed. See `src/features/opening_hours.py`.

## Sensor matching

Each sensor takes the nearest segment within 30 m, with two adjustments:
- A road centreline tagged `sidewalk=separate` is skipped, because the sensor stands on one of the separately mapped footpaths.
- Crossings get a 5 m penalty, because sensors sit on the footpath beside them.

A sensor is flagged for review when:
- the match is over 15 m away
- the runner-up is within 3 m of the pick
- it matched a road centreline
- the segment runs across the sensor's counting direction (`direction_1`/`direction_2`, over 30° off)
- it is an indoor sensor

**To hand-check:** open `sensor_match_review.csv` (up to 3 candidates per flagged sensor) next to a map. Put the right `segment_id` in `configs/sensor_overrides.yaml` under `melbourne:` as `location_id: segment_id`, then re-run `--only sensors`. Overrides apply to one extract only, so re-check them when the extract changes.

## Caveats

- **Sensors without locations.** 18 sensor IDs in the 2009–2022 archive are not in the current sensor locations file (e.g., 7 Birrarung Marr, 13 Flagstaff Station, 15 State Library). They hold 17.5% of archive rows. They get no OSM features until someone finds their coordinates.
- **The snapshot is from 2026.** OSM shows today's network and businesses, applied to counts back to 2009. That is unavoidable for a portable model, and only the Metro Tunnel stations are handled explicitly. Shops that opened or closed during the record are not.
- **Sidewalk mapping varies.** Some streets have separate footpath ways, others only a centreline. This affects which segment a sensor lands on and the network metrics. The sensor review covers the first issue; `sidewalk_*` flags let a model learn the second.
- **Betweenness favours unique shortcuts.** The top segments for both betweenness and angular choice at 1,600 m are the Melbourne General Cemetery's avenues: long, straight paths that are the only short route across a big block. The CBD grid splits its through-movement across many parallel streets. The measures are working as intended here. Use the radius and weighted variants together rather than one global ranking.
- **Edge effects.** The study bbox has a ~500 m buffer around the City of Melbourne. Metrics at 1,600 m are truncated for segments near the bbox edge, though not for the CBD sensors.
- **Licence.** OSM is ODbL. These outputs are a derived database: fine to use inside the project and for model outputs (credit "© OpenStreetMap contributors"), but distributing the tables themselves would bring share-alike obligations. The metrics use only BSD-licensed code (numba, scipy). We did not use cityseer because it is AGPL-3.0 and the what-if API may need to recompute metrics at serve time.
