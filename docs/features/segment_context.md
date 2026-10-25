# Shade, events and surroundings

Features that describe what is around a segment and what is happening there, added on top of the OSM tags and network metrics in `docs/features/osm_features.md`. Each has a portable form any city can build, and some have a Melbourne-rich form from City of Melbourne layers (CLAUDE.md, "Portable vs. Melbourne-rich model"); `src/features/schema.py` marks which is which.

## Shade (`src/features/shade.py`, stage `shade`)

For each position of the sun the site sees in a year (azimuth in 30-degree steps, elevation in bands from 5 to 90 degrees), every building footprint is swept away from the sun by height ÷ tan(elevation), and points every 5 m along each segment are tested against the shadows. The result is `osm_segment_shade.parquet`: `segment_id`, `sun_bin` and the shaded fraction in two variants.

| Variant | Heights | Notes |
|---|---|---|
| `shade_portable` | OSM: `height`, else `building:levels` × 3 m, else 9 m | The 9 m default is an assumption; in Melbourne 71% of OSM buildings use it (`shade_meta.json` counts them) |
| `shade_rich` | The city's own: CoM 2023 footprints (`footprint_extrusion`), City of Boston (`BLDG_HGT_2010`, feet) | Empty where a site has no such layer |

The training frame looks each hour up by its sun bin (`sun_elevation_deg`, `sun_azimuth_deg`); with the sun below 5 degrees a segment counts as fully shaded. The sweep uses each footprint's convex hull with its moved copy, so a concave building slightly overstates its shadow. Trees aren't in it; canopy cover (below) is a separate feature.

## Events (`src/features/events.py`)

Sources: AFL games at the MCG and Marvel Stadium (`data/raw/events/afl_attendance.csv`) and the sourced log in `configs/events.yaml` (cricket, concerts, football, Halloween). The segment stage writes each segment's walking distance to each venue (`osm_segment_venue_dist.parquet`, up to 3 km); the training frame adds, per row:

| Column | Meaning |
|---|---|
| `event_pre`, `event_during`, `event_post` | An event near the segment starts within 2 h, is on, or ended within 2 h |
| `event_dist_m` | Walking distance to the nearest venue with an event in any of those phases (3,000 when none) |
| `event_day_dist_m` | The same for any event that day, including those logged with a date only |

The columns are generic: there is never one per venue, which would amount to a location ID. Where a source gives no end time, a default length per kind applies (`events.default_hours`; AFL 3 h, Test and Shield days 7 h, T20 and BBL 3.5 h through the log's `duration_h`). Crowd sizes aren't used: the log has none, and AFL Tables' attendance is known only after a game, so at prediction time it would have to be a user's labelled estimate.

## Surroundings (`src/features/surroundings.py`)

| Column | Variant | Meaning |
|---|---|---|
| `wc_tree_share`, `wc_grass_share`, `wc_built_share` | Portable | Share of ESA WorldCover 2021 pixels (10 m) of each class in a window about 50 m around the segment. Too coarse for the path itself; it describes the setting |
| `dem_slope_mean_pct`, `dem_slope_max_pct` | Portable | Grade along the segment, sampled every 10 m from the site's DEM, so the GA 5 m, USGS 1 m and USGS 10 m DEMs are measured the same way. The CoM footpath survey slope (`slope_pct`, per sensor) stays Melbourne-only |
| `canopy_cover` | Melbourne | Share of a 25 m buffer under the CoM 2021 tree canopy |
| `clue_jobs_400`, `clue_jobs_800` | Melbourne | Jobs in CLUE blocks (latest census year) within 400 or 800 m along the network |
| `businesses_200`, `businesses_400` | Melbourne | Business establishments (latest year) within 200 or 400 m along the network |

Like the OSM snapshot, the CLUE and business counts are one recent year applied to the whole count record.
