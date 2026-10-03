# Project Context

Full background for the project: what it is, why decisions were made, and what's in and out of scope. `CLAUDE.md` holds the rules and conventions; this file holds the reasoning. When a task touches something here, read the relevant section first. Items marked **VERIFY** have not been confirmed; do not build on them until someone checks.

---

## 1. The product

**Name:** Stoa (renamed from DesireLine, October 2026). Earlier candidates: DesireLine, Sightline, Footfall, Wayward, Strider, Agora. Avoid "Pathfinder" (an existing crowd-simulation product).

**Problem:** cities, campuses, and event organizers change public spaces (new buildings, closed paths, event layouts) while guessing how people on foot will react. They find out afterward: worn dirt shortcuts across lawns, bottleneck sidewalks, ignored entrances or booths. Counting people is expensive, so most places have little data.

**What it does:** predicts where people walk, how many, and when, on every path of a site, and lets a planner test changes before making them. It accounts for slope, shade and heat, ice, what people can see ahead, nearby destinations and whether they're open, events, and informal shortcuts missing from maps.

**Users:**
- **City planners and transportation agencies (primary market).** Need network-wide pedestrian volumes for safety and investment decisions; already pay for pedestrian data.
- **Campus planners.** Wedge market and case study; small on its own.
- **Event organizers.** Only the outdoor approach network around a venue (streets, transit links). Venue interiors belong to crowd-simulation incumbents.
- **Researchers and heritage-site teams.** A generic research mode with standard GIS outputs (e.g., archaeologists studying movement through ancient sites). Adoption and validation, not revenue.

---

## 2. Competitive landscape

| Group | Examples | What they do | Where they stop |
|---|---|---|---|
| Crowd microsimulation | MassMotion (Oasys), Legion, PTV Viswalk, Pathfinder, AnyLogic, STEPS, Pedestrian Dynamics, SimWalk; open source SUMO, GAMA | Agent-based simulation of stadiums, stations, airports, events | Specialist-run, built around venue geometry, behavior hand-calibrated |
| Pedestrian data vendors | StreetLight, Replica, Placer.ai, Strava Metro | Estimate existing volumes, mostly from phone location data | Can't evaluate unbuilt designs; StreetLight's link-level pedestrian data was disrupted by 2022 smartphone privacy changes; Strava Metro limited to annual totals by privacy masking |
| Configurational analysis | Space Syntax (consultancy), depthmapX | Predict movement potential from street-network structure | Fixed metrics, not learned; consultancy delivery; static (no time, weather, events) |

**Our differentiation:**
- Learns behavior per site from data instead of hand-set formulas.
- Works on designs that don't exist yet.
- Self-serve, not consultancy or specialist software.
- Outdoor, network-scale, climate-aware (shade, heat, ice, slope, visibility).
- No phone tracking: built from maps, imagery, and a small number of counts.
- Can be complementary to microsimulation: a calibrated demand and route-choice layer that feeds those tools.

**Positioning against Space Syntax:** not "we replace it" but "we build on it." Space Syntax metrics are model *features*. Space Syntax predicts static movement potential; planners need time-, weather-, and event-dependent volumes, which require fitting to data.

**Supporting research:** network-wide pedestrian volume estimation is active research but not an established product (Portland 2026 paper on network-wide estimation; Tel Aviv study finding gradient boosting promising; Melbourne study using a graph-diffusion recurrent model on the city's sensor data).

---

## 3. Case studies

| Site | Role | Notes |
|---|---|---|
| **Melbourne** | Training and validation; all accuracy claims rest here | Open hourly sensor counts since 2009, with direction |
| **NEU Boston campus** | Demonstration (no local ground truth) | Dense, mostly flat; desire paths across quads |
| **Event scenario** | Demonstration | Melbourne event days for validation; a stadium or campus event for the demo |
| NEU Oakland campus | Stretch: transfer test | Hilly; tests whether the model generalizes |
| Fenway game day | Stretch: partial real check | See section 6 |
| NEU Fall Fest | Stretch: booth-exposure analysis | See section 9 |

**Constraint:** manually counting people at events or on campus is off-limits. Local ground truth must come from existing data sources or from customers.

---

## 4. Data

### 4.1 Melbourne pedestrian counts (primary ground truth)

- **Hourly counts:** `https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/pedestrian-counting-system-monthly-counts-per-hour/exports/csv`
- **Sensor locations:** `https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/pedestrian-counting-system-sensor-locations/exports/csv`
- **Minute-level past-hour feed:** related dataset, useful for live monitoring later.
- Use the City of Melbourne portal, not mirrors (the state mirror is stale).

**Format (current export):** semicolon-delimited.
```
id;location_id;sensing_date;hourday;direction_1;direction_2;pedestriancount;sensor_name;location
107220261003;107;2026-10-03;2;9;5;14;280Will_T;-37.81246271, 144.95690188
```
- `pedestriancount = direction_1 + direction_2`.
- What direction 1 and 2 mean is defined **per sensor** in the Sensor Locations file (`direction_1`, `direction_2` columns; likely compass labels, **VERIFY**). Never assume a global convention. Join on `location_id`.
- `hourday` is 0-23 (0 = midnight to 1am).
- `location` is one "lat, lon" string.
- **VERIFY** how far back direction values go; older rows may have totals only.

**Quirks:**
- A zero means nobody passed, but long runs of zeros at a normally busy sensor are outages, not empty streets. Detect and mask them.
- Sensors are added, moved, renamed, and retired. Use IDs, track each sensor's active period (installation date and status are in Sensor Locations), and never evaluate a sensor on dates it didn't exist.
- Older exports were wide (one column per sensor); current exports are long. Normalize to long.
- Full history is large (an older snapshot was ~418 MB).

**License:** Sensor Locations is CC BY 3.0 AU (on data.gov.au). Hourly counts listed as "other-open" on the state mirror. **VERIFY** the counts license on the city portal's information page before use.

### 4.2 Other sources

| Dataset | Role | License |
|---|---|---|
| OpenStreetMap | Network, footways, amenities, opening hours | ODbL: share-alike on distributed derived databases (matters for research-mode exports) |
| USGS 3DEP 1 m lidar DEM | Terrain (US) | Public domain |
| NAIP imagery (~0.6-1 m) | Path segmentation (US) | Public domain |
| MassGIS aerial imagery | Possibly sharper imagery for NEU | **VERIFY** resolution and license |
| Copernicus GLO-30 DEM (prefer over SRTM) | Terrain (global) | Free; check attribution |
| Sentinel-2 | Surface, vegetation | Copernicus open license |
| ESA WorldCover, Dynamic World | Land cover | CC BY 4.0 (attribution) |
| Building footprints with heights | Shade where no lidar | OSM `height`/`building:levels`, open building datasets; **VERIFY** per region |
| Melbourne terrain/buildings | Training-site resolution | **VERIFY**: Australian public lidar coverage; city building footprints with heights |
| Weather (Melbourne, hourly) | Features | **TBD source** |
| Event calendars | Event features | Public schedules; manual curation |
| Melbourne amenity datasets (businesses, toilets, fountains) | Destinations | **VERIFY** availability |
| Stanford Drone Dataset | Route-choice validation only | **Non-commercial: never train on it** |
| ETH/UCY, DUT, HC | Route-choice validation only | Treat as validation-only unless explicit commercial permission found |

### 4.3 Possible additional sources (stretch)
- **Strava Metro academic program:** free one-year access to aggregated pedestrian/cycling data. **Applications close October 23, 2026**; notification in November, so likely useful after the course. Skews toward runners/recreational walkers. Academic terms: validation only.
- **Advan (formerly SafeGraph Patterns) via Dewey:** weekly visit counts per point of interest (US/Canada, since 2017). **VERIFY** whether NEU has access. Academic terms: validation and attraction weighting in research, not product training.
- **Open counts from other cities** for cross-city transfer (NYC DOT pedestrian counts likely; Dublin, Zurich possibly). **VERIFY** availability and licenses.
- **MBTA gated station entries** (`https://mbta-massdot.opendata.arcgis.com/`): **VERIFY** existence, time resolution, and license.

### 4.4 Licensing rules
- Non-commercial data: validation only, never product training.
- CC BY-SA / ODbL: share-alike obligations; get advice before distributing derived databases or models trained on share-alike data.
- No personal data, no phone tracking. Customer-uploaded data stays private to that customer's site.

---

## 5. Features

All features are per path segment (and per hour where time-varying). The model never sees pixels.

- **Network structure:** shortest-path betweenness; Space Syntax-style integration and angular choice (e.g., cityseer or momepy on an OSMnx graph), computed within limited radii.
- **Terrain:** slope along the segment.
- **Shade and thermal comfort:** shaded fraction per hour/season, computed geometrically from building heights or lidar surfaces; seasonal behavior (shade sought in summer; sun sought and ice avoided in winter).
- **Surface and greenery:** from land cover / imagery.
- **Visibility:** what's visible from the segment (viewshed engine), because people walk toward what they can see.
- **Destinations and amenities:** counts of shops, restaurants/cafés, toilets, fountains, benches within short *network* walking distances; "open now" per hour from opening hours; weather interactions (fountains, shade, toilets matter more on hot days).
- **Time:** hour, day of week, holidays.
- **Weather:** temperature, rain, etc.
- **Events:** distance to and timing of events.

**Amenity caveat:** shops locate where foot traffic already is (reverse causality). Fine for predicting current flows; what-ifs that add/remove amenities may overstate effects and must be labeled rough guidance.

### 5.1 Resolution tiers

| Feature | High-res tier (US: 1 m lidar, 0.6-1 m imagery) | Global tier (~10-30 m) |
|---|---|---|
| Slope | Reliable at footpath scale | Smoothed; global DEMs include buildings/trees, so urban slope is rough |
| Shade | Lidar surface or building heights | Building footprints with heights; DEM can't provide it |
| Surface/greenery | Fine detail | Usable at block scale |
| Informal path detection | Possible | **Not possible** (1-2 m paths invisible at 10 m); OSM paths only |

**Pipeline mechanics:** reproject to a local metric CRS (UTM); resample continuous data with interpolation and categorical data with nearest/majority (never interpolate classes); summarize along segment buffers with fractional pixel-coverage weighting (e.g., `exactextract`); store source, native resolution, and acquisition date with every feature.

**Train/serve resolution mismatch:** the model trains in Melbourne. If deployed where only coarse data exists, inputs differ from training. Mitigations: compute Melbourne features at both resolutions and train with degraded copies (or a model per tier); evaluate the accuracy gap honestly (high-res vs. coarse-only on held-out sensors); include training examples with features missing; flag predictions as lower confidence when a site relies on the coarse tier.

---

## 6. Models and components

### 6.1 Flow model (core)
- Predicts hourly pedestrian volume per segment, with uncertainty bounds.
- Start with gradient boosting or regularized regression; GNNs must prove themselves against it. Data is dozens to low hundreds of sensors, not thousands.
- **Baseline ladder** (each rung shows what the next adds): (1) shortest-path betweenness, (2) Space Syntax metrics + calibration, (3) hand-weighted costs, (4) learned model with all features.
- **Why learning is needed at all:** static metrics can't express time variation, interactions (heat at noon in summer; ice only below freezing), or events.

### 6.2 Path segmentation
- Fine-tune a pretrained segmentation model on campus aerial imagery, using OSM footways as rough labels; detected informal paths are added to the graph.
- Campus scale only. Paths are 2-3 pixels at NAIP resolution; canopy hides many. OSM footways are often a few meters off the imagery, so labels are noisy; validate on a hand-checked sample.
- Justified on product grounds: OSM misses informal paths.

### 6.3 Agents
**Which agents are built is TBD; at least one is required.**
- **LLM intake agent:** turns a planner's natural-language description (or an uploaded event/traffic plan) into a structured event spec. Asks about missing or ambiguous fields; labeled defaults otherwise; every field records its source. Ambiguous places go to the map for click-to-confirm, never silently guessed.
- **LLM planning assistant:** tool-calling over deterministic functions (edit graph, rerun model, compare scenarios) and narrates the differences. Never computes numbers.
- **Simulated walkers:** sample routes from the model's route probabilities over the graph and aggregate per segment and time step to show congestion and animate crowds. Amenities act as destinations, weighted by attraction and opening hours. Not a social-force microsimulator.

Example intake:
> "Saturday's game at 7:10, gates open 90 minutes before. Sellout expected. Most fans come by the Green Line to Kenmore; Jersey Street closed to cars; food trucks on Lansdowne."

becomes a spec with date, start, gates_open, attendance ("sellout" needs venue capacity, so ask), arrival modes, closures, and vendors, each with a source, followed by map confirmation.

### 6.4 Viewshed engine
- Existing 3D, aperture-aware GPU ray-casting engine (PyTorch), validated at 97-99% cell agreement against GRASS `r.viewshed`.
- Uses: visibility as a model feature; sequential "what you see as you walk" views along routes ("serial vision" in urban design); booth/table visibility for event layouts.
- Heavy at city scale: limit to demo areas, precompute, cache.

---

## 7. Validation

- **Split by location, not time:** hold out whole sensors/areas; image tiles split by area.
- **Macro (flow volume):** held-out Melbourne sensors; must beat the Space Syntax baseline (which typically correlates ~60-80% with observed movement). Headline result: "Space Syntax explains X; adding climate, visibility, and destinations gets Y more."
- **Event days:** evaluated separately; build a calendar of major Melbourne events (e.g., White Night appears in the data; also New Year's Eve, Moomba, AFL Grand Final and parade, Australian Open, Christmas events). Check which fall near active sensors.
- **Calibration curve:** calibrate on k sensors (3, 5, 10, 20), predict the rest, plot accuracy vs. k. Demonstrates "calibrate with a few counts" without manual counting, and exercises the retrain loop.
- **Micro (route choice):** held-out Stanford Drone scenes; displacement and route-overlap metrics. Validation only.
- **Synthetic parameter recovery:** plant a known preference in simulated walkers and check the model recovers it. Tests the method, not real behavior; say so.
- **Sanity scenarios:** e.g., closing a path lowers its flow and raises flow on alternatives.
- **Segmentation:** held-out tiles; recall of known paths.
- **Intake agent:** field-level extraction accuracy and location placement accuracy on an evaluation set of descriptions with correct specs (start building this early).
- **Rule:** simulations validate the pipeline, never the model's realism.
- Boston/NEU outputs are demonstrations, not validations: "validated in Melbourne, deployed to Boston."

**Fenway (stretch):** game-day MBTA Kenmore entry surge minus same-weekday baseline approximates the walking crowd from the ballpark to the station (some bus-transfer leakage). Public attendance gives total departures; the Kenmore share is one exit stream; the remainder went to other stops, commuter rail, rideshare, or walking. Mass conservation gives a partial check on simulated exit splits. **VERIFY** MBTA data first.

---

## 8. MLOps lifecycle

- **Train:** DVC-versioned snapshots of counts, OSM, imagery, weather, events; MLflow experiment tracking with data version per run.
- **Package:** model + feature pipeline (same code as training) + viewshed engine in one Docker image; MLflow registry with model card.
- **Validate:** location-based splits; beat baselines and the deployed model; no event-day regression; sanity scenarios; segmentation and agent evals.
- **Deploy:** Cloud Run API + web app; canary with automatic rollback; per-site calibrated variants stored separately.
- **Monitor:** accuracy against each month's new Melbourne counts (real delayed ground truth); input drift (unusual weather, new event types); prediction drift; new OSM path edits as segmentation labels; agent correction rates, grounding violations, and cost per conversation; resolution tier of each deployment.
- **Retrain:** monthly on new counts; triggered by accuracy drop or drift; per site on new customer counts; all retrained models re-run the gates.
- **Expo:** a small in-app panel showing model version, last retrain, latest accuracy on new Melbourne data, and drift status.

**Infrastructure:** GCP Cloud Run, Cloud Storage, Artifact Registry; GitHub Actions CI/CD. Orchestration TBD: Cloud Composer is likely too expensive for the budget; prefer Cloud Scheduler + Cloud Run jobs or Airflow on a small VM. Segmentation training on free/university GPUs (Colab, Kaggle, NEU cluster), not the GCP budget. Keep one Cloud Run instance warm during the expo.

**Cost to serve:** precompute heavy work; GNN/GBM inference is cheap; simulation runs per scenario, not per click; the LLM narrates finished results and isn't called per map interaction.

---

## 9. Onboarding flows

### Event planner (stadium)
1. **Mark the site** on a map (venue + surrounding blocks). Network, terrain, buildings, transit stops, amenities, imagery (path detection), and the weather forecast load automatically.
2. **Describe the event in natural language** (intake agent). Required: date, gate-opening and start/end times, expected attendance, gates/exits. Optional: arrival mode split, parking and rideshare locations, closures and barriers, fan zones, food trucks, merchandise.
3. **Add existing data (optional):** ticket scan timestamps from past events (most valuable), turnstile counts, transit ridership on past event days.
4. **Review and test:** predicted flows through arrival, mid-event, and post-event surge; hotspots; what-ifs (close a street, open an exit, move trucks, stagger release); assistant explains differences.
5. **After the event:** upload actual gate scans; recalibrates for next time.

Caveats shown to the user: the post-event surge is the hardest to predict; this is planning guidance, not safety certification.

### Fall Fest booth exposure (stretch case study)
- Data on hand: 564 table placements across 11 zones (Robinson Quad 47, Sculpture Park 11, Library Quad 47, World Series Way 45, Cabot Pathway and Quad 52, Krentzman Quad 46, Egan Pathway 29, Centennial Common 52, JDOAAI Quad 43, West Village Quad 87, West Campus 105). Zones and table numbers only, no coordinates.
- Output: exposure score per table = predicted passersby x visibility from approach paths.
- Needs: physical table layout from the Center for Student Involvement (request it, along with stage/food/entrance locations, timing, and any aggregate attendance; offer a post-event layout analysis in return). Fallback: assume table numbers run in physical sequence, stated as an assumption.
- Placement is themed (e.g., Greek life in Robinson Quad), confounding location with club type; compare positions **within** zones.
- No outcome data yet; any club sign-up counts must be voluntarily shared.
- Commercial link: trade shows price booths by location.

---

## 10. Bottlenecks and risks

**Top bottlenecks:**
1. **Matching sensors to graph edges.** Sensors sit on specific sidewalks; OSM often has single street centerlines. Hand-check every sensor early; version the mapping. Direction labels help.
2. **Interactive what-if speed.** Network metrics are global; recompute only within a radius or the affected neighborhood; keep demo areas small; precompute demo scenarios.
3. **Segment IDs changing.** Stable IDs; add edges rather than renumber; versioned graph referenced by every artifact.
4. **Single integrator.** One person owns integration and the viewshed engine; document interfaces early and pair someone on integration.

**Other:** noisy segmentation labels; time alignment (time zones, Melbourne daylight saving: use one timezone-aware convention); event calendar curation; shade and viewshed compute at scale; agent eval set takes time; place-name ambiguity; data/graph module blocks everyone; Cloud Run cold starts; live demo failure (precompute, recorded fallback).

**Project risks:** learned model may not beat Space Syntax (checkpoint below); Melbourne license unverified; only one city of ground truth; scope creep.

**Checkpoint (weeks 5-6):** if the learned model doesn't beat the Space Syntax baseline, stop investing in the model and pitch the product as Space Syntax made interactive, 3D, and climate-aware. The what-if tool, walking viewshed, and climate-aware routing stand on their own.

---

## 11. Scope

### Tier 1: Immediate scope (absolute deliverables)
1. Melbourne data pipeline: counts (with direction), sensor locations, outage masking, sensor active periods, OSM graph, hand-checked sensor-to-edge matching; DVC versioning.
2. Feature pipeline: network metrics, slope, shade, surface/greenery, amenities with opening hours, weather, time, events; visibility as a precomputed per-segment feature.
3. Flow model: gradient boosting baseline-ladder comparison, location-based splits, event-day evaluation, calibration curve.
4. Aerial path segmentation at NEU campus scale, feeding detected paths into the graph (fallback: OSM paths only).
5. Agents: at least one, type TBD (LLM intake/planning assistant and/or walker sampling), with its evaluation set.
6. Event days: Melbourne event calendar, event features, separate evaluation.
7. Web app: map, predicted flows, one or more what-if interactions; NEU Boston campus demonstration.
8. Full MLOps loop: tracking, registry, validation gates, CI/CD, Cloud Run deployment, monitoring against new monthly counts, scheduled and triggered retraining, expo status panel.
9. Data cards, license documentation, model cards.

### Tier 2: Stretch goals (if time and budget allow, roughly in priority order)
1. Sequential "what you see as you walk" viewshed visualization along routes.
2. Second agent type (whichever wasn't chosen in Tier 1).
3. Document upload for intake (event plans, traffic management PDFs).
4. Per-site calibration upload flow in the UI (customer counts, ticket scans).
5. Fenway game-day case study (MBTA surge + attendance check).
6. Synthetic parameter-recovery test.
7. Cross-city transfer test (train Melbourne, test another city's open counts).
8. Global-tier support: degraded-resolution training and the measured accuracy gap.
9. NEU Oakland campus transfer demonstration.
10. Fall Fest booth-exposure analysis (needs CSI table map).
11. Advan visit-weighted amenity attraction (if NEU has Dewey access).
12. Strava Metro validation (if the application is accepted).
13. Street-level surface/shade classification from open street-level imagery (e.g., Mapillary; CC BY-SA, so share-alike caution).

### Tier 3: Out of scope (future work; do not start without a team decision)
- Inverse reinforcement learning for route preferences (the upgrade path once the GBM baseline exists).
- Social-force or full microsimulation.
- Venue interiors.
- Crowd-safety certification or guarantees of any kind.
- Real-time streaming predictions.
- Native mobile apps.
- Global production coverage.
- Formal causal modeling of amenity effects.

### Explicitly rejected (never)
- Phone or individual-level tracking.
- Training the product model on non-commercial datasets.
- LLM-computed or LLM-estimated numbers, routes, or locations.
- Manual counting of people by the team.

---

## 12. Business model

- **Tiers:** campus/site license (one site, a few seats); city license (whole network, scenario builder, API); research mode (free or discounted, open GIS exports).
- **Unit economics:** cost per scenario stays low by design (precomputation, cheap inference, LLM only narrates finished results), so no per-query cost can outgrow the subscription price.

---

## 13. Open decisions and verification to-dos

**Decisions:** which agents; graph and feature-table formats; orchestration tool; weather source; success-criteria margins; expo date and timeline.

**Verify:** Melbourne counts license; direction value history; Melbourne lidar and building-height availability; MassGIS imagery resolution/license; Melbourne amenity datasets; MBTA gated entries; NEU Dewey access; other cities' open counts; Cloud Composer pricing; competitor feature claims (do microsimulation tools support outdoor/climate features?) before stating them in a pitch.

**Deadlines:** Strava Metro academic application closes **October 23, 2026**.
