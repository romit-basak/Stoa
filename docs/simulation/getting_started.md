# Simulation: getting started

How to build the simulation on the data we already have: route-sampling walkers first, then JuPedSim for one outdoor area. The design and the reasons behind it are in [modelling_plan.md](../models/modelling_plan.md), "Simulation". This page covers what to build, in what order, and which files to read. Nothing in `src/simulation/` exists yet.

## The two layers

1. **Walkers (our code, `src/simulation/walkers.py`)** decide who goes where and when: origins, destinations, routes on the street graph, and departure minutes. They work on the whole graph and are cheap.
2. **JuPedSim (`src/simulation/microsim.py`)** moves those people through one small outdoor area, second by second, with walking speed falling as density rises. It takes its entries, timings and routes from the walkers.

The flow model doesn't run the simulation. It sets the targets the walkers are fitted to, and the sensors check the result.

## What to read

| Data | File | Use |
|---|---|---|
| Street graph | `data/interim/osm/melbourne/graph/edges.parquet` (111,465 segments; `segment_id`, `u`, `v`, OSM tags, geometry) and `routing_edges.parquet` | Routes for the walkers, and the walkable area's geometry for JuPedSim |
| Sensor positions | `data/processed/features/melbourne/sensor_segment_map.parquet` (`location_id` → `segment_id`) | Where the counts sit on the graph |
| Hourly counts | `data/processed/features/melbourne/counts_hourly.parquet` (one row per sensor-hour, with `direction_1` and `direction_2` since Oct 2024) | Targets and checks for each hour at the sensors |
| Hourly predictions | Flow model, `stoa-flow-rich` in MLflow (`docs/models/flow_model.md`) | Targets on segments without a sensor. Predicting arbitrary segments isn't built yet; until it is, fit to the sensors' observed counts |
| Within-hour profiles | `minute_profiles.parquet` (per sensor) and `minute_profiles_by_tier.parquet` (busy, medium or quiet; for places without a sensor). Columns: `day_type`, `hour_local`, `hour_type` (ordinary, event_pre, event_during, event_post), `block_start` (0 to 55), `share` | Departure minutes within each hour |
| Peakiness | `minute_hours.parquet`, column `peak15` (the busiest quarter's count over a quarter of the hour) | How bunched arrivals are, for the random draws |
| Raw per-minute counts | `gs://stoa-dvc-data/minute_feed/minute_feed_<date>.parquet` (from 25 Oct 2026; `dvc pull` or the pipeline's `pull_minute_feed` brings them into `data/raw`) | Validation, mainly event egress |
| Events | `data/processed/training/melbourne/event_days.parquet`, `configs/events.yaml` (times and venues), `osm_segment_venue_dist.parquet` | Which days and hours are event phases, and how far each segment is from a venue |

The minute counts come in 5-minute blocks: 57 of 99 sensors only publish one row per 5 minutes, and eight per-minute sensors never publish minutes :00 to :04 (listed in `minute_profiles_meta.json`). Work in 5-minute blocks, and leave those eight sensors out of minute-level checks. The archive is missing 11 November after 04:34 and all of 12 November.

## Walkers, step by step

1. **Origins and destinations.** Tram and train stops, car parks and entrances as origins; shops, cafés, offices and venues as destinations, weighted by attraction and whether they're open that hour (OSM `opening_hours`). For an event, the venue's gates are the origin after the show.
2. **Route choice.** Shortest paths on a cost of distance, plus slope, shade and crossings, with some randomness so walkers don't all take one path. Several routes per origin and destination pair.
3. **Fit the trips to the counts.** Each route crosses some segments, so the simulated count on a segment is the sum of the trips whose routes cross it. Solve for non-negative trip numbers that make those sums match the target counts, starting from a prior such as a gravity model, because many trip tables fit the same counts. Iterative proportional fitting, or non-negative least squares with weights, both work. Segments with wide intervals count less. The toy example in the modelling plan shows the idea on four segments.
4. **Departure minutes.** Spread each hour's trips over its 5-minute blocks with the profile for that place, day type and hour type. Draw them at random rather than using the average share: real hours are bunchier than any average. A peak quarter is 1.32 on ordinary hours and 1.43 after events, while the average profiles give 1.14. A test on 13 November found a learned minute model no better than the pooled profiles, so use the profiles.
5. **Directions.** Split each segment's flow with the sensor's `direction_1` and `direction_2` shares.
6. **Check.** Simulated counts should match the targets they were fitted to. Real sensors left out of the fit are the actual test.

What-ifs (close a path, add an entrance) edit the graph, re-route and refit. The deploy check, where closing a path lowers its flow and raises flow on the alternatives, should fall out of the walkers, since they keep the number of people the same.

## JuPedSim, step by step

- **Install:** `pip install jupedsim` (1.4.2, LGPL-3.0, wheels for Python 3.10 to 3.14 on macOS, Linux x86_64 and Windows). It also pulls in PySide6 and VTK, about 280 MB, so it runs in its own container, not the API or Airflow image. No Linux ARM wheel exists, so in Docker on a Mac set `platform: linux/amd64`, or run it natively in the venv.
- **First area: Melbourne Park after a concert**, not Marvel Stadium. The archive has about ten concerts at Rod Laver Arena since 25 October. Five sensors between the arenas and the city (161, 136, 123, 124 and 75) catch every crowd leaving, almost all walking one way. At sensor 161 on 27 October, 1,417 people passed between 22:50 and 23:30. Marvel's events in the archive are 11 November (lost) and 14 November, with more from 27 November. Use Marvel as the second area.
- **Geometry:** buffer the area's footpath segments into a walkable polygon with shapely, and cut out buildings. Outdoor approach areas only, never inside a venue.
- **Agents:** each walker entering the area becomes an agent at its entry point at its departure time, with its route as waypoints and an exit. Start with the Collision Free Speed Model, the simplest and quickest. Social Force can come later.
- **Measurements:** put a measurement line at each sensor's position, and count crossings per 5-minute block and direction. That gives series to compare with the per-minute archive. PedPy, from the same Jülich group, computes flow and density from trajectories; check its licence before adding it.

## Validation

| Check | Against |
|---|---|
| Peak timing, peak height and the delay between sensors along the route | Per-minute counts on concert nights. Fit departure spread and walking speed on some nights; test on the others |
| Speed against density, flow through bottlenecks | Jülich controlled experiments (availability and licence still to confirm) |
| Route choice | Stanford Drone trajectories: validation only, nothing fitted to it ships (CC BY-NC-SA 3.0, core rule 5) |

Report these separately from the flow model's accuracy. The sensors only show when and how many people pass fixed points, not where everyone is at once.

## Rules that apply

- Results are planning guidance for outdoor approach areas, never a crowd-safety assessment (core rule 7). This matters most for the microsimulation, because its outputs look like safety results.
- Every number shown comes from the models or the simulation, never from the LLM (core rule 1).
- No individual-level data: the walkers are synthetic, and the sensors count totals (core rule 6).

## Suggested order

1. Walkers on the Melbourne Park area, fitted to the observed hourly counts at its sensors on one concert night.
2. Departure minutes from the profiles, then compare the walkers' 5-minute crossings at sensor 161 with the archive.
3. The same area in JuPedSim with the walkers as input, and the same comparison.
4. Calibrate departure spread and speed across concert nights; then Marvel.
5. Switch the targets to flow-model predictions once segment prediction exists.
