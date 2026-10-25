# Data pipeline: gaps and changes needed

Review of 24 October 2026. It covers the `stoa_pipeline` DAG, the graph and feature code, and what the pipeline still needs before it can feed training, validation and deployment end to end. Facts were checked against the code, the data on disk and the live City of Melbourne portal on that date. Tier numbers refer to `docs/PROJECT_CONTEXT.md` section 11.

## Status on the extension branch (25 October 2026)

The work below was done on `ext/pipeline` (plan: data side and training frame; the model, API and walkers are outside it). Each gap's state:

| Gap | State | Where |
|---|---|---|
| No site build in the DAG | Done: `stoa_site_build` for NEU Boston and NYC | `dags/stoa_site_build.py`, `docs/pipeline/site_build.md` |
| Sites are hand-written config | Still five places, now checked to agree by a test; a bbox-to-site resolver is Tier 2 | `tests/test_site_config.py` |
| OSM source is city-level | Deferred (Tier 2) | |
| DEM is per-country code | Done for the three sites: the source is site config, large USGS clips come in tiles, no source fails loudly | `src/ingest/dem.py` |
| No generic buildings with heights | Partly: shade has a portable variant from OSM heights (9 m default, counted) and a rich one from city heights | `src/features/shade.py` |
| Imagery is US-only and single-item | Deferred (Tier 2) | |
| AFL fixtures not in the DAG | Event features use AFL Tables (already pulled) and the sourced event log; Squiggle still runs by hand | `src/features/events.py`, `configs/events.yaml` |
| No entrance nodes | Done, in a separate routing graph so segment IDs don't change | `src/features/graph.py` |
| Stations are single points | Done: station entrances are rail access points, dated like their station | `src/features/osm_extract.py` |
| Shops are counts, not destinations | Deferred: belongs to the walkers | |
| Areas aren't walkable | Done: edges across plazas | `src/features/graph.py` |
| Roads double-counted | Done: separate-sidewalk centrelines left out of the metrics, bridging ones kept | `src/features/graph.py` |
| Unmapped crossings | Done: inferred at unmarked junctions, flagged `is_inferred` | `src/features/graph.py` |
| No informal paths | Hook only: `detected_paths.parquet` joins the graph once segmentation writes it | `src/features/graph.py` |
| Sensor matches not hand-checked | Skipped by the team; the training frame records the count and has `--drop-needs-review` | `src/features/training_frame.py` |
| No shade | Done | `src/features/shade.py` |
| No sun azimuth | Done | `src/features/sun.py` |
| No surface or greenery features | Done: WorldCover shares, CoM canopy; surface stays the OSM tag | `src/features/surroundings.py` |
| No event features | Done | `src/features/events.py` |
| No visibility feature | Deferred: needs the viewshed engine | |
| Melbourne-rich layers unused | Done for CLUE jobs, businesses and canopy; CLUE floor space isn't held | `src/features/surroundings.py` |
| No feature schema | Done | `src/features/schema.py` |
| Leak trap left to the user | Done: the training frame applies it, tested | `src/features/training_frame.py` |
| Bucket protection unknown | Soft delete (7 days) is on; object versioning is off and needs an owner to turn on | README |
| Irreplaceable data has one remote | Open: where the second copy lives is a team decision | |
| `.dvc` pointers committed by hand | Still by hand: committing from Airflow needs a deploy key, a team decision | |
| No drift against training data | Done: PSI per feature against the training frame's reference | `src/pipeline/validate.py` |
| No accuracy monitoring | Deferred: needs a model | |
| No OSM change monitoring | Done: weekly `stoa_osm_watch` | `src/pipeline/osm_watch.py` |
| Documentation points | Done | |

Also added, from the follow-up requests: the per-minute counts archive (`stoa_minute_feed`, the Explorer cluster poller, checks in `validate`), the Stanford Drone annotations (validation only), CI, and the NYC feature mask.

## Where things stand

The pipeline runs for Melbourne only. One daily Airflow DAG pulls the public sources, builds the OSM walking graph and segment features, joins the hourly training table (7,409,868 rows, 118 sensors, May 2009 to now), validates it, runs bias slicing and pushes the result to DVC. The NEU Boston and NYC data are one-off downloads outside the DAG.

Nothing downstream of the data exists yet: there is no `src/models/`, `src/api/`, `src/viewshed/`, `src/simulation/` or `app/`, no MLflow tracking, and no `.github/workflows/`. The only Dockerfile builds the Airflow image.

### The DAG

| Task | Source and API | Output |
|---|---|---|
| `pull_counts` | City of Melbourne Opendatasoft Explore API v2.1: live hourly table and sensor locations. On a first run, also the 2009–2022 archive attachment and the Internet Archive copy of the 2025-05-02 export | `data/raw/melbourne_portal/` (live table merged on location, date and hour) |
| `pull_weather` | Open-Meteo archive API (ERA5), hourly, city centre | `data/raw/weather/`, one file per year |
| `pull_calendars` | `holidays` package, Victorian Government "Important Dates" CSV, `configs/calendars/` | `data/raw/calendars/` |
| `pull_static_layers` | Portal (footpath steepness, CLUE), Geoscience Australia WCS (5 m DEM), AFL Tables season pages | Fetched only when missing |
| `pull_osm` | BBBike weekly extract | `data/raw/osm/melbourne.osm.pbf`, pinned |
| `osm_extract`, `osm_graph`, `osm_segment_features`, `osm_sensor_match` | Local | `data/interim/osm/melbourne/`, `data/processed/features/melbourne/` |
| `sensor_steepness` | Local | `sensor_steepness.parquet` |
| `join_all` | Local | `counts_hourly.parquet` |
| `validate` | Great Expectations | `data/processed/validation/melbourne/` |
| `bias_slices`, `bias_errors` | Local | `data/processed/bias/` |
| `dvc_push` | Google Cloud Storage, `gs://stoa-dvc-data` | New data version |

Every download records URL, SHA-256, retrieval time and licence in a `MANIFEST.json`; every processing stage writes a `*_meta.json` with input hashes, row counts and runtimes.

## Clarifications to the plan

### Three pipelines, not one

The work splits into three pipelines with different inputs:

1. The training pipeline (Melbourne, the current DAG). Its output is the training table.
2. A site build for each site the app serves. It pulls public data for a bbox, builds the graph, adds detected paths and computes segment features. It needs no counts. Most of the graph build is deterministic code; the one learned piece is path segmentation, which adds informal paths as edges.
3. Prediction. The trained model runs on a site's segment features for each hour, per-site calibration rescales it with any local counts, and scenario edits are applied on top.

Calibration data belongs to prediction, not to the site build. User counts set the absolute scale of the predictions; they don't change the graph.

### The flow model is a direct-demand model

The Tier 1 gradient-boosted model predicts each segment's hourly count from the segment's context (network metrics, amenities, transit, slope) and the hour's weather and calendar. It routes nobody. Route choice enters only through betweenness and angular choice, which assume shortest or least-angle paths.

Section 6.5 says walkers "sample routes from the flow model's route probabilities". A direct-demand model has no route probabilities. The walkers need their own route-choice model: origins and destinations (entrances, stations, amenities) and a cost function over the graph, tuned so that walker flows summed per segment match the flow model. This needs a team decision before the walker work starts.

### Hourly resolution

Hourly is enough for the flow model. A sensor counts everyone who passes in the hour, whatever they do before or after (board a bus, enter a shop, go home), and the model predicts that number. Melbourne's history is hourly, NYC's counts are 2–3 hour totals, and a walk across a site takes minutes.

Hourly falls short in three places:

- Peaks shorter than an hour (class changes, event egress, train arrivals) are averaged out. Crowding depends on the peak. Walkers and microsimulation need time steps of a few minutes, with the hourly total spread by a within-hour profile labelled as an assumption.
- Hourly counts on neighbouring segments don't add up, because people enter and leave the network at buildings and stations. The flow model doesn't need them to balance; walkers need explicit sources and sinks.
- There is no minute-level history. The portal's per-minute past-hour feed has no archive, so sub-hour validation is only possible from data the project collects from now on.

### Melbourne counts arrive daily

The portal's dataset description says "updated on a monthly basis". The data says otherwise: on 24 October, 2026-10-23 had a full day (2,308 rows), 2026-10-24 already had 343, and `data_processed` was 15:39 UTC that day. The description was last edited on 2024-08-14 and also says directional counts aren't included, which is no longer true.

Weather (ERA5) also updates daily, about 5–7 days behind. Calendars change yearly, by hand. OSM is pinned on purpose. A daily run is justified by counts and weather arriving daily, and by validation catching a dead sensor or stale feed within a day; it costs about 6 minutes. The README's stated reason ("so no day of the counts' rolling window is missed") doesn't hold: each run re-pulls from two days before the newest held date, so a missed run is filled by the next one.

### Storage and what can't be rebuilt

The DVC remote is a GCS bucket. A clean clone rebuilds almost everything from public sources in about 20 minutes, and every teammate's machine holds `data/` and a DVC cache. A GCS outage would stop `dvc_push`, not lose data.

Three things exist only in our copies:

- Count days that have left the portal's rolling window. The live table now starts on 2024-10-25; our copy starts on 2024-10-04. The Wayback copy covers up to 2025-05-02, so from May 2027 every day that rolls off exists nowhere public.
- The pinned OSM extract (2026-09-26). BBBike serves only the current extract, and segment IDs and sensor overrides are tied to this one.
- The validation history (`counts_hourly_stats_history.jsonl`), which the drift checks compare against.

The more likely risk is administrative: course credits ending, a closed billing account or an accidental delete.

### What "drift" means today

The drift checks in `validate` compare each table with the previous good run: rows didn't drop by more than 1%, sensors didn't disappear, missing shares and the mean count stayed close. They catch broken data. They don't compare against the data a model was trained on, and they say nothing about concept drift, because there is no model.

Melbourne makes concept drift measurable directly, because true counts arrive daily. COVID (`covid_period`) and the Metro Tunnel opening (`post_metro_tunnel`, `recent_station_dist_m`) are known breaks already handled as features. At sites with no counts, such as NEU Boston, only input and prediction drift can be watched.

### Site onboarding versus Tier 1

Tier 1 needs site builds for NEU Boston (the demo, item 9) and NYC (the cross-city test, item 4). Both already have site entries and can use the existing `--site` path. A general "give a bbox, get a site" flow for any user goes beyond that and overlaps with global-tier support (Tier 2, item 12). It shouldn't delay the Tier 1 site builds.

### Surface costs and shortcuts

A desire path forms where the distance saved outweighs the extra cost of the worse surface, so route choice minimises a generalised cost: length times surface and slope factors. Melbourne's sensors can't teach that cost. Of 119 outdoor sensors, 82 are on sidewalks and about 28 on formal off-street paths (bridges, riverside promenades, parks, Bourke Street Mall); none is on a crossing, a lawn or an informal path. By surface, 102 are on paved segments, 16 are untagged and one is on fine gravel.

Ways to ground the costs, in the order they could be done:

1. Energy-cost priors. Pandolf et al. (1977), using Soule and Goldman's terrain coefficients: blacktop 1.0, dirt road 1.1, light brush 1.2, heavy brush 1.5, swampy bog 1.8, loose sand 2.1. Tobler's hiking function for slope, with his off-path speed factor of about 0.6. These measure effort, so they understate the penalty people act on (wet shoes, mud, "keep off the grass").
2. Desire paths as revealed preference. For each lawn bordered by paths, compute the shortcut a worn path would offer, label whether one is visible in the imagery, and fit the probability that a path exists against the distance saved, with betweenness as a traffic control (trails need enough walkers to form; Helbing et al., *Nature* 1997). The break-even point is the penalty people actually act on. MassGIS 15 cm leaf-off imagery covers NEU, Harvard, MIT, BU and Tufts, and all of it is public domain.
3. Trajectory validation. The Stanford Drone Dataset has real campus routes across lawns. Under core rule 5 it should check weights obtained elsewhere, not be their source.
4. Conditional factors (wet grass after rain, unlit paths at night) can't be learned from any Stoa data. They fit the core rule 11 pattern: fixed levels the user picks, shown as assumptions.

The weights would enter as baseline rung 3 ("hand-weighted costs"), as cost-weighted betweenness features, and above all as the walkers' route-choice cost. Research mode's plan to compare cost hypotheses, with route ensembles and null models, uses the same weights.

## Gaps found

### Ingest and site build

| Gap | Detail | Tier |
|---|---|---|
| No site build in the DAG | NEU Boston and NYC are hand-run downloads; their OSM features have never been built or checked | 1 |
| Sites are hand-written config | A site needs entries in five places: `sites`, `osm.extracts`, `worldcover.tiles`, `calendars.holidays` in `configs/data.yaml`, and `sites` in `configs/features.yaml` | 2 (general onboarding) |
| OSM source is city-level | BBBike covers a fixed list of cities; Boston already needed the Cambridge extract | 2 |
| DEM is per-country code | GA 5 m for Melbourne, USGS 3DEP for US sites with `utm_epsg`, otherwise fixed 30 m tiles | 2 |
| No generic buildings with heights | CoM and Boston have city datasets; elsewhere only OSM, where about 4% of Melbourne buildings carry heights | 2 |
| Imagery is US-only and single-item | NAIP and MassGIS only; fails when one NAIP item doesn't cover the bbox | 1 for NEU (works), 2 elsewhere |
| AFL fixtures not in the DAG | `src.ingest.events` (Squiggle) runs by hand; only AFL Tables crowds are pulled, and only for bias slicing | 1 |

### Graph

| Gap | Detail | Tier |
|---|---|---|
| No entrance nodes | CLAUDE.md's interface says nodes are intersections and entrances; only intersections are built. `entrance=*` and `railway=subway_entrance` aren't read | 1 (walkers, store siting) |
| Stations are single points | Transit is only a network distance from the nearest node | 1 |
| Shops are counts, not destinations | POIs snap to the nearest node within 50 m and become amenity counts | 1 (walkers) |
| Areas aren't walkable | Pedestrian plazas are dropped from the graph (kept only as `ped_area_dist_m`); squares and quads can't be crossed | 1 |
| Roads double-counted | 6,626 road centrelines tagged `sidewalk=separate` stay in the graph beside their mapped sidewalks; sensor matching skips them, but betweenness and choice route through them | 1 |
| Unmapped crossings | Only OSM-mapped crossings exist; no sensor is on a crossing, so crossing volumes are always extrapolated | 1 |
| No informal paths | `src/models/segmentation/` doesn't exist; the NEU imagery is pulled | 1 |
| Sensor matches not hand-checked | 52 outdoor and 34 indoor sensors are flagged `needs_review`; `configs/sensor_overrides.yaml` is empty | 1 |

### Features

| Gap | Detail | Tier |
|---|---|---|
| No shade | Nothing computes shadows. CoM 2023 footprints (41,701, all with heights), tree canopies and trees are pulled but unused; the 2018 DSM is only fetched with `--large` | 1 |
| No sun azimuth | The hourly table has `sun_elevation_deg` only | 1 (needed for shade) |
| No surface or greenery features beyond OSM tags | `surface` is tagged on 61% of footway and path edges; WorldCover is pulled but unused, and at 10 m is too coarse at path scale | 1 |
| No event features | No AFL or event columns in the hourly table | 1 |
| No visibility feature | `src/viewshed/` doesn't exist; the engine itself exists as public GSoC work on 3D viewshed analysis | 1 |
| Melbourne-rich layers unused | CLUE jobs and floor space, business establishments, canopy, CoM pedestrian network | 1 |
| No feature schema | `src/features/schema.py` is still TBD, and no column is marked portable or Melbourne-only, which Tier 1 item 2 asks for | 1 |
| Leak trap left to the user | `recent_station_dist_m` must be capped before 2025-11-30 by whoever builds the training frame | 1 |

### Storage, validation and monitoring

| Gap | Detail | Tier |
|---|---|---|
| Bucket protection unknown | Object versioning and soft delete weren't checked (`gcloud` not installed on the reviewing machine) | 1 |
| Irreplaceable data has one remote | Rolled-off count days, the pinned OSM extract and the validation history | 1 |
| `.dvc` pointers committed by hand | `dvc_push` uploads, but a person commits `data/*.dvc` and `dvc.lock` | 1 |
| No drift against training data | Validation compares run to run only | 1 |
| No accuracy monitoring | No model yet; the monthly accuracy check is planned only | 1 |
| No OSM change monitoring | The pinned extract is never compared with the latest one | 1 |

### Documentation

- `docs/pipeline/validation.md` says `source` must be `archive` or `live`; `configs/validation.yaml` also allows `wayback`, which is correct.
- The `bias_errors` runtime is given as about 10 minutes (DAG comment), 14 minutes (README, first-run timing) and 18 minutes (README and CLAUDE.md).
- The README's reason for the daily schedule is the rolling window; the real reason is that counts and weather arrive daily (see above).
- The Melbourne counts data card should note that the portal's "monthly" description is out of date.

## Changes needed

### End to end: site build

1. Add a `stoa_site_build` DAG, triggered per site with a `site` parameter: pull OSM, DEM, WorldCover, weather, calendars and, where they exist, buildings and imagery, then run the OSM feature stages. Run it for `neu_boston` and `nyc` and check the outputs the way Melbourne was checked. (Tier 1)
2. Graph changes, each with a test:
   - add entrance nodes from `entrance=*`, and station entrances from `railway=subway_entrance` and `public_transport=stop_position`, connected to the nearest walkable edge;
   - make pedestrian areas walkable, with edges across each area between its boundary access points;
   - exclude or down-weight road centrelines tagged `sidewalk=separate` when computing network metrics;
   - add inferred crossings at junctions where both sides have sidewalks and no crossing is mapped, flagged as inferred. (Tier 1)
3. Path segmentation for NEU: train on MassGIS 2025 leaf-off imagery with OSM footways as rough labels, add detected paths as edges with `source = detected` and a surface class, and check recall on a hand-labelled sample. (Tier 1)
4. Later, a site resolver that turns a bbox into a full site entry (centre, timezone, UTM CRS, WorldCover tiles, holiday country), plus Overpass or Geofabrik in place of BBBike, and global fallbacks (Copernicus 30 m DEM, open building footprints), recording the resolution tier each site got. (Tier 2)

### Training

1. Build the missing Tier 1 features:
   - shade: shaded fraction per segment for a grid of sun positions, from CoM footprints (Melbourne-rich) and OSM or city heights (portable), plus a `sun_azimuth_deg` column in `counts_hourly`;
   - surface and greenery per segment;
   - event features from AFL fixtures and crowds (bring `src.ingest.events` into the DAG);
   - a precomputed visibility feature from the viewshed engine;
   - the Melbourne-rich layers (CLUE, businesses, canopy).
2. Write `src/features/schema.py` for the feature table: column types, the source of each column, and a portable or Melbourne-only mark. Generate `configs/schemas/counts_hourly.yaml` from it.
3. Add a training-frame builder (for example `src/models/flow/dataset.py`). It filters to outdoor, located sensors without suspected outages, joins segment features and opening hours, caps `recent_station_dist_m` before 2025-11-30, picks the portable or Melbourne-rich columns, and applies the location-based splits (held-out sensors and the precinct holdouts in `configs/bias.yaml`). This keeps the leak traps in one tested place.
4. Hand-check the 52 flagged outdoor sensors and fill `configs/sensor_overrides.yaml` before the first model run.
5. Implement the baseline ladder (shortest-path betweenness; Space Syntax plus calibration; hand-weighted costs with the surface and slope factors above) and the gradient-boosted model, with MLflow runs that record the DVC data version and the OSM extract SHA-256.
6. Decide on the walker route-choice model (see "Open decisions") before building walkers.

### Validation

1. Model gates in code: beat the baselines and the deployed model on held-out sensors (at least 15% lower MAE on log counts than Space Syntax plus calibration), 80% intervals covering 75–85%, no event-day regression, and the sanity scenarios.
2. Transfer tests: NYC rank correlation and log-total error before and after calibrating on 5 locations (needs the NYC site build); within-Melbourne precinct holdouts, uncalibrated and calibrated on 1–2 sensors.
3. The calibration curve: accuracy against the number of local sensors used.
4. Event days separately, centred on Marvel Stadium.
5. Bias step 4: rerun `bias_errors` on the real model.
6. Shortcut costs: fit the desire-path threshold on Boston-area campus imagery and check routes against the Stanford Drone Dataset, without fitting anything shipped to it.
7. Fix the documentation points above.

### Deployment and monitoring

1. CI with GitHub Actions: tests, lint, and the DAG import test on every PR; build and deploy on merge to main.
2. An API image, separate from the Airflow image, holding the model, the same feature code as training, and per-site artifacts keyed by site and OSM extract SHA-256. Precompute baseline flows per site. Scenario edits (closing a path) change network metrics, so the API needs a local recompute of features around the edited segments instead of the full 2.5-minute build.
3. Monitoring in the DAG:
   - a monthly job that scores the deployed model on the new Melbourne counts at held-out sensors (concept drift) and triggers retraining on an accuracy drop;
   - feature drift (PSI or KS per feature) against the snapshot the deployed model was trained on;
   - prediction drift per site, the only signal for NEU Boston;
   - a weekly comparison of the pinned OSM extract with the latest one, reporting path edits near sensors.
4. Storage: turn on object versioning and soft delete on `gs://stoa-dvc-data`, keep a second copy of the raw counts, the pinned OSM extract and the validation history outside GCS, commit the `.dvc` pointers from the pipeline or on a schedule, and plan for the bucket before course credits end.
5. Optional and cheap: archive the portal's per-minute past-hour feed from now on, for later sub-hour validation.

## Open decisions for the team

1. The walker route-choice model: origins and destinations, cost function, and how walker flows are matched to the flow model's segment totals.
2. Whether cost weights fitted on the Stanford Drone Dataset count as a model "fine-tuned on" it under core rule 5. This review assumes they do.
3. How plazas and lawns are represented: edges across areas, detected paths only, or a grid over open areas (the Tier 2 terrain-grid builder).
4. Whether inferred crossings go in the graph, and how they're shown.
5. Where the second copy of irreplaceable data lives.
6. Whether to archive the per-minute feed.
