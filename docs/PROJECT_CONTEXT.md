# Project Context

Full background for the project: what it is, why decisions were made, and what's in and out of scope. `CLAUDE.md` holds the rules and conventions; this file holds the reasoning. When a task touches something here, read the relevant section first. Items marked **VERIFY** have not been confirmed; do not build on them until someone checks.

---

## 1. The product

**Name:** Stoa (renamed from DesireLine, October 2026). Earlier candidates: DesireLine, Sightline, Footfall, Wayward, Strider, Agora. Avoid "Pathfinder" (an existing crowd-simulation product).

**Problem:** cities, campuses, and event organizers change public spaces (new buildings, closed paths, event layouts) while guessing how people on foot will react. They find out afterward: worn dirt shortcuts across lawns, bottleneck sidewalks, ignored entrances or booths. Counting people is expensive, so most places have little data.

**What it does:** predicts where people walk, how many, and when, on every path of a site, and lets a planner test changes before making them. It accounts for slope, shade and heat, what people can see ahead, nearby destinations and whether they're open, events, and informal shortcuts missing from maps. Ice is handled by an explicit, labeled assumption rather than learned (see section 7.1).

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
- Outdoor, network-scale, climate-aware (shade, heat, slope, visibility; ice only as a labeled assumption).
- No phone tracking: built from maps, imagery, and a small number of counts.
- Can be complementary to microsimulation: a calibrated demand and route-choice layer that feeds those tools.

**Positioning against Space Syntax:** not "we replace it" but "we build on it." Space Syntax metrics are model *features*. Space Syntax predicts static movement potential; planners need time-, weather-, and event-dependent volumes, which require fitting to data.

**Supporting research:** network-wide pedestrian volume estimation is active research but not an established product (Portland 2026 paper on network-wide estimation; Tel Aviv study finding gradient boosting promising; Melbourne study using a graph-diffusion recurrent model on the city's sensor data).

---

## 3. Case studies

| Site | Role | Notes |
|---|---|---|
| **Melbourne** | Training and validation; all accuracy claims rest here | Open hourly sensor counts since 2009, with direction |
| **Melbourne transfer precincts** | Within-city transfer test | Areas held out whole and unlike the training areas: University of Melbourne edge (Swanston St sensors 42-44), Carlton (Lygon St), North Melbourne (Errol St, Macaulay Rd). See section 7.1 |
| **New York City** (NYC DOT bi-annual counts) | Cross-city transfer test | Train on Melbourne, predict NYC's 114 count locations. The only evidence for any claim beyond Melbourne. Data checked; see 4.3 |
| **NEU Boston campus** | Demonstration (no local ground truth) | Dense, mostly flat; desire paths across quads. Not validated: Boston outputs rest on the transfer tests above, not on Boston data |
| **Event scenario** | Demonstration | Melbourne event days for validation; a stadium or campus event for the demo |
| NEU Oakland campus | Stretch: transfer demonstration | Hilly. No local counts, so it's a demonstration, not a test |
| Fenway game day | Stretch: partial real check | See section 6 |
| NEU Fall Fest | Stretch: booth-exposure analysis | See section 9 |

**Constraint:** manually counting people at events or on campus is off-limits. Local ground truth must come from existing data sources or from customers.

---

## 4. Data

### 4.1 Melbourne pedestrian counts (primary ground truth)

- **Hourly counts:** `https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/pedestrian-counting-system-monthly-counts-per-hour/exports/csv`. **The live table is a rolling ~2-year window** (2024-10-04 → now when first pulled); history before Nov 2022 is a separate archive attachment, and **Nov 2022 – Oct 2024 is in neither**.
- **Sensor locations:** `https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/pedestrian-counting-system-sensor-locations/exports/csv`
- **Minute-level past-hour feed:** related dataset, useful for live monitoring later.
- Use the City of Melbourne portal, not mirrors (the state mirror is stale).

**Format (current export):** semicolon-delimited.
```
id;location_id;sensing_date;hourday;direction_1;direction_2;pedestriancount;sensor_name;location
107220261003;107;2026-10-03;2;9;5;14;280Will_T;-37.81246271, 144.95690188
```
- `pedestriancount = direction_1 + direction_2`.
- What direction 1 and 2 mean is defined **per sensor** in the Sensor Locations file (`direction_1`, `direction_2` columns: 53 sensors North/South, 47 East/West). Never assume a global convention. Join on `location_id`.
- `hourday` is 0-23 (0 = midnight to 1am).
- `location` is one "lat, lon" string.
- Directional counts exist only in the live window (from 2024-10-04). The 2009–2022 archive has totals only.

**Quirks:**
- Outages appear as **missing rows**, not zeros (zeros are ~0.007% of rows). Never impute missing hours as zero; still check for long zero runs at busy sensors.
- Sensors are added, moved, renamed, and retired. Use IDs, track each sensor's active period (installation date and status are in Sensor Locations), and never evaluate a sensor on dates it didn't exist.
- Older exports were wide (one column per sensor); current exports are long. Normalize to long.
- Full history is large (the archive is ~420 MB unzipped).
- **Re-pulls must never overwrite.** The ingest merges each pull of the rolling table into the rows already held (keyed on `id`); the first pull is also kept as a dated snapshot. DVC versions every pull once set up.
- **Usable archive is smaller than "2009–2022" suggests:** 24% of archive rows fall in the 2020–21 COVID period, and only 43 sensors have 5+ non-COVID years. North Melbourne's archive sensors start in 2020–22, so that precinct has almost no normal pre-gap history.
- **Metro Tunnel (opened 2025-11-30; full timetable 2026-02-01; Town Hall's Swanston St entrance 2026-01-29)** is a structural break. Sensors within 300 m of the new stations rose a median ~4.5% year on year (others ~flat); sensor 42 (Grattan St, UniMelb edge) fell ~23%. The 2019 CoM network has no station entrances: add them with opening dates, and report before/after separately or with a station-access feature.
- Feature layers are snapshots (CoM network 2019, building footprints mostly 2018) applied to counts from 2009; CLUE (annual since 2002) carries year-by-year change in jobs, floor space and businesses.

**License:** CC BY 4.0 under the portal-wide terms ("Our open data is provided under a creative commons licence … free to share and adapt our data for any purpose, provided you give the appropriate credit", About our data page; confirmed 2026-10-04). The dataset's own licence field is empty; every other CoM dataset lists CC BY 4.0.

### 4.2 Other sources

| Dataset | Role | License |
|---|---|---|
| OpenStreetMap | Network, footways, amenities, opening hours | ODbL: share-alike on distributed derived databases (matters for research-mode exports) |
| USGS 3DEP 1 m lidar DEM | Terrain (US) | Public domain |
| NAIP imagery | Path segmentation fallback (US). NEU: 2023, 0.3 m, **leaf-on** (pulled) | Public domain |
| MassGIS aerial imagery | **Primary** path-segmentation input for NEU: 2025, 15 cm, **leaf-off**, so paths under trees are visible (pulled) | Public domain; credit MassGIS |
| City of Boston buildings with roof breaks | Building heights for NEU shade/viewshed (pulled; heights from a 2010 survey, in feet; 5.4% missing) | PDDL |
| Copernicus GLO-30 DEM (prefer over SRTM) | Terrain (global) | Free; check attribution |
| Sentinel-2 | Surface, vegetation | Copernicus open license |
| ESA WorldCover, Dynamic World | Land cover | CC BY 4.0 (attribution) |
| Building footprints with heights | Shade where no lidar | OSM `height`/`building:levels` (sparse: ~4% of Melbourne buildings), city datasets per region |
| Melbourne terrain/buildings | Training-site resolution | GA 5 m lidar DEM (CC BY 4.0; land voids stored as 0) + CoM footpath-steepness survey; CoM 2023 footprints with heights. Open 1 m terrain doesn't exist for Melbourne (Vicmap/FABDEM excluded on licence) |
| Weather (hourly, all sites) | Features | ERA5 via Open-Meteo (proposed default; pulled for Melbourne, NEU, NYC). CC BY 4.0; free API non-commercial |
| Event calendars | Event features | AFL fixtures (Squiggle) + crowds (AFL Tables, facts); other events hand-curated with a source per row |
| Holidays and academic calendars | Time features | `holidays` package (MIT); Vic school terms (CC BY 4.0); UniMelb and NEU dates entered by hand (UniMelb terms forbid scraping) |
| Melbourne amenity datasets | Destinations | Available (pulled, CC BY 4.0): businesses and cafés 2002–2024, 74 toilets, 338 fountains. **None has opening hours** |
| Stanford Drone Dataset | Route-choice validation; fine-tuning for demonstration only, never shipped | **CC BY-NC-SA 3.0** (applies to any mirror). Never in a shipped model's training data; don't distribute demo weights fine-tuned on it |
| ETH/UCY, DUT, HC | Route-choice validation only | Treat as validation-only unless explicit commercial permission found |

### 4.3 Possible additional sources (stretch)
- **Strava Metro academic program:** free one-year access to aggregated pedestrian/cycling data. **Applications close October 23, 2026**; notification in November, so likely useful after the course. Skews toward runners/recreational walkers. Academic terms: validation only.
- **Advan (formerly SafeGraph Patterns) via Dewey:** weekly visit counts per point of interest (US/Canada, since 2017). **VERIFY** whether NEU has access. Academic terms: validation and attraction weighting in research, not product training.
- **Open counts from other cities** for the cross-city transfer test (now Tier 1; see section 7.1).
  - **Primary:** NYC DOT Bi-Annual Pedestrian Counts (`cqsj-cfgu` on NYC Open Data, checked 2026-10-04): 114 locations (100 on-street, mostly retail corridors; 13 bridges; Hudson River Greenway), counted twice a year (May and Sept/Oct) on one weekday (07-09, 16-19) and the adjacent Saturday (12-14). Values are **period totals, not hourly**; 0 means not collected. Coverage May 2007 to May 2026, with no rounds from May 2019 to Oct 2020. Oct 2025 weekday medians are ~560/h (AM) and ~1,230/h (PM). Compare each total with the sum of predicted hours over the same window.
    - Data: `https://data.cityofnewyork.us/Transportation/Bi-Annual-Pedestrian-Counts/cqsj-cfgu` (API `https://data.cityofnewyork.us/resource/cqsj-cfgu.json`); methodology: `https://www.nyc.gov/html/dot/downloads/pdf/bi-annual-ped-count-readme.pdf`.
    - License: NYC Open Data Law (Admin Code §23-502(d)): no registration, license requirement or use restrictions; name the source and version.
  - **Secondary:** NYC DOT *Bicycle and Pedestrian Counts* (`ct66-47at`, sensors in `6up2-gnw8`): continuous 15-minute automated counts, updated daily, but only four pedestrian sites (Willis Ave bridge, High Bridge, Emmons Ave, Concrete Plant Park), all bridges or waterfront paths. Useful only as a check of hourly profiles. Each site is recorded under two sensor IDs with identical counts (keep one), and only Emmons Ave and Concrete Plant Park still report.
  - Dublin and Zurich: not checked.
- **MBTA gated station entries** (`https://mbta-massdot.opendata.arcgis.com/`): **VERIFY** existence, time resolution, and license.

### 4.4 Licensing rules
- Non-commercial data: validation, plus fine-tuning for a demonstration that is never shipped. Never in a shipped model's training data. Share-alike NC data (e.g., Stanford Drone): don't distribute anything derived from it, including demo weights.
- A mirror can't relicense a dataset: the original licence applies to every copy, whatever the mirror's label says.
- CC BY-SA / ODbL: share-alike obligations; get advice before distributing derived databases or models trained on share-alike data.
- No personal data, no phone tracking. Customer-uploaded data stays private to that customer's site.

---

## 5. Features

All features are per path segment (and per hour where time-varying). The model never sees pixels.

**Portable vs. Melbourne-only features.** Several of the strongest Melbourne inputs exist only there: CoM footpath-steepness survey, CLUE jobs and floor space, business establishments, canopy polygons, the CoM pedestrian network. A model trained on them can't run in NYC or Boston. So train two variants: a **Melbourne-rich** model (headline Melbourne accuracy) and a **portable** model on features any city has (OSM network and amenities, terrain, land cover, weather, building heights, calendars). The portable model is the one used for the cross-city test and the Boston demo; report the accuracy gap between them in Melbourne.

**Transferability rule:** every feature must mean the same thing in Boston as in Melbourne. Use physical quantities (temperature, sun elevation, daylight hours) instead of calendar month, because Melbourne's summer is December to February, so "January = hot" is wrong in Boston. Never use sensor IDs, location IDs, or raw coordinates as features: they let the model memorize places instead of learning patterns, and they mean nothing at a new site. Section 7.1 has the full reasoning.

- **Network structure:** shortest-path betweenness; Space Syntax-style integration and angular choice (e.g., cityseer or momepy on an OSMnx graph), computed within limited radii.
- **Terrain:** slope along the segment.
- **Shade and thermal comfort:** shaded fraction per hour/season, computed geometrically from building heights or lidar surfaces; seasonal behavior driven by temperature and sun position (shade sought when hot, sun sought when cold). Ice is not a learned feature (see 7.1).
- **Surface and greenery:** from land cover / imagery.
- **Visibility:** what's visible from the segment (viewshed engine), because people walk toward what they can see.
- **Destinations and amenities:** counts of shops, restaurants/cafés, toilets, fountains, benches within short *network* walking distances; "open now" per hour from opening hours; weather interactions (fountains, shade, toilets matter more on hot days). **Opening hours are sparse:** ~1,800 OSM features in the study area carry `opening_hours` against ~15,000 amenities/shops (~12%), and the CoM datasets have none. Treat "open now" as a partial feature with explicit missingness.
- **Time:** hour, day of week, public holidays (local to each site), sun elevation, daylight hours. No calendar month or day of year.
- **Academic term:** in-semester flag for segments near a university, from the published academic calendar (University of Melbourne in training; NEU at serve time). Justified by data: Melbourne sensors 42-44 at the university's edge run at ~0.45x their mean in January and December and ~1.6x in March, while CBD sensors barely change. This captures semester rhythm only, not class-change surges (see 7.1).
- **Weather:** temperature, rain, etc.
- **Events:** distance to and timing of events; crowd size where known (AFL attendance).
- **Transit access:** distance to stations and stops, with opening dates (Metro Tunnel stations from 2025-11-30).

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
- **Why learning is needed at all:** static metrics can't express time variation, interactions (heat at noon in summer; rain with no shelter), or events.
- **Shape vs. scale:** the general model learns relative patterns (which segments and hours are busier than others); per-site calibration sets the absolute scale from local counts. Calibration is the core of the generalization strategy, not an add-on (see 7.1).

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
- **Event days:** evaluated separately. **Marvel Stadium carries the AFL event-day evaluation**: 7 sensors within 500 m, reporting on every game day. The MCG has no sensor within 500 m; its four nearest (0.7–1 km) include two of the least complete sensors, and in the archive only 93 of 655 MCG game days have one reporting. Other major events (White Night, NYE, Moomba, Grand Final parade, Australian Open) go in a hand-curated list with a source per row.
- **Calibration curve:** calibrate on k sensors (3, 5, 10, 20), predict the rest, plot accuracy vs. k. Demonstrates "calibrate with a few counts" without manual counting, and exercises the retrain loop. The full curve runs on Melbourne overall and on NYC's 114 locations; the transfer precincts are too small for it (3, 5 and 6 sensors), so there calibrate on 1–2 sensors and test the rest. This is expected to be the project's strongest result.
- **Micro (route choice):** held-out Stanford Drone scenes; displacement and route-overlap metrics. Validation (and demo-only fine-tuning, never shipped). Needs only the trajectory annotations (~450 MB), not the videos.
- **Synthetic parameter recovery:** plant a known preference in simulated walkers and check the model recovers it. Tests the method, not real behavior; say so.
- **Sanity scenarios:** e.g., closing a path lowers its flow and raises flow on alternatives.
- **Segmentation:** held-out tiles; recall of known paths.
- **Intake agent:** field-level extraction accuracy and location placement accuracy on an evaluation set of descriptions with correct specs (start building this early).
- **Rule:** simulations validate the pipeline, never the model's realism.
- Boston/NEU outputs are demonstrations, not validations: "validated in Melbourne, deployed to Boston."

### 7.1 Generalization and transfer

Boston is demonstration only, so held-out Melbourne sensors alone can't tell us whether anything carries over to a new site. Some of what the model learns should transfer and some almost certainly won't. These tests measure which is which.

**Expected to transfer reasonably well:**
- **Network structure.** The link between street-network configuration and movement is the most replicated finding in this field, observed across many cities. Which segments are *relatively* busy should carry over.
- **Physical effects.** Slope, shade, and temperature affect walkers through the same mechanics everywhere, even if tolerance varies.
- **Destination effects.** Higher traffic near transit, shops, and open amenities holds broadly.

**Expected not to transfer:**
- **Absolute volumes.** A Boston campus path and a Melbourne CBD street differ by orders of magnitude. The model may rank segments sensibly and still get the numbers badly wrong.
- **Ice and snow.** Melbourne's central city essentially never freezes, so Melbourne data can't teach anything about ice. Any ice effect in Boston is an assumption.
- **Campus rhythms.** Campus traffic surges between classes and is quiet during them. Hourly CBD data can't teach that. Semester-level rhythm is learnable (see the academic-term feature in section 5), but class-change surges are not.
- **Sensor placement bias.** Melbourne put sensors where traffic matters. In the current window the median sensor's median hour is ~210 people; only about a quarter of sensors are below ~100/h, and only a handful (a Birrarung Marr bridge, a park entrance, a Docklands walkway) are below 20/h. Quiet paths, which make up much of a campus, are thin in the training data.
- **Hemisphere.** Seasons are reversed (handled by the transferability rule in section 5).

**What we do about it:**
1. **Transferable features only** (section 5): physical quantities instead of calendar month; no sensor or location identifiers; the **portable** model variant (no Melbourne-only layers) for any test or demo outside Melbourne.
2. **Separate shape from scale** (section 6.1): the general model learns relative patterns; per-site calibration from local counts sets the scale.
3. **Within-Melbourne transfer test.** In addition to the grouped K-fold over sensors, hold out whole precincts whose character differs from the training areas. Train on the CBD; test on:
   - **University of Melbourne edge:** sensors 42 (Grattan St-Swanston St), 43 (Monash Rd-Swanston St), 44 (Tin Alley-Swanston St), with history since 2015. These sit on Swanston St at the campus boundary, not on interior campus paths, so they're campus-*like*, not campus. Medians 56-87/h.
   - **Carlton:** Lygon St (31, 37, 50), Pelham St (46), Lincoln-Swanston (54).
   - **North Melbourne / Kensington:** Errol St (70, 87), Queensberry-Errol (86), Macaulay Rd (76, 85, 180).
   Report each precinct's error uncalibrated and after calibrating on 1–2 of its sensors (the precincts are too small for the full curve). The UniMelb-edge sensors also sit in the Metro Tunnel's Parkville catchment, so split their results at 2025-11-30.
4. **Cross-city transfer test (Tier 1, even if small).** Train on Melbourne; predict NYC DOT's bi-annual counts (section 4.3), comparing each period total with the summed predicted hours. Never train on them. This is the only evidence for any claim beyond Melbourne. Report rank correlation (does it get *which* places are busy right?) separately from absolute error (does it get *how many*?), before and after calibrating on a few local counts.
5. **Ice: rule or omit.** Either apply an explicit, hand-set rule (e.g., a penalty on unsheltered, sloped segments below freezing with recent precipitation) labeled in the UI as an assumption, or leave ice out of Boston predictions. Never present it as learned. Team decision pending.
6. **Name the campus gap.** For this semester, campus predictions don't capture class-change surges; class schedules would need their own data source. The UI and any write-up say so plainly.

**How to present it:** don't claim the model generalizes globally. The claim is:
- the model learns pedestrian patterns in Melbourne;
- the within-Melbourne transfer test shows how well it handles unfamiliar kinds of areas;
- the cross-city test shows what happens in a new city;
- calibration with a small number of local counts closes much of the remaining gap (the calibration curve).

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
1. **Matching sensors to graph edges.** Sensors sit on specific sidewalks; OSM often has single street centerlines. Against the CoM network it's feasible (every outdoor sensor is within 18 m of a walkable link, median 2 m), but each sensor has ~3 candidate links within 10 m (both footpaths plus crossings). Hand-check every sensor early; version the mapping. Direction labels help.
2. **Interactive what-if speed.** Network metrics are global; recompute only within a radius or the affected neighborhood; keep demo areas small; precompute demo scenarios.
3. **Segment IDs changing.** Stable IDs; add edges rather than renumber; versioned graph referenced by every artifact.
4. **Single integrator.** One person owns integration and the viewshed engine; document interfaces early and pair someone on integration.

**Other:** noisy segmentation labels; time alignment (time zones, Melbourne daylight saving: use one timezone-aware convention); event calendar curation; shade and viewshed compute at scale; agent eval set takes time; place-name ambiguity; data/graph module blocks everyone; Cloud Run cold starts; live demo failure (precompute, recorded fallback).

**Project risks:** learned model may not beat Space Syntax (checkpoint below); only one city of dense ground truth (partly addressed by the within-Melbourne and NYC transfer tests, section 7.1); model transfers patterns but not volumes (calibration addresses this); the portable model may be much weaker than the Melbourne-rich one; Metro Tunnel break in the most recent counts; losing count history to the rolling window (mitigated by merge-only ingest and DVC); scope creep.

**Checkpoint (weeks 5-6):** if the learned model doesn't beat the Space Syntax baseline, stop investing in the model and pitch the product as Space Syntax made interactive, 3D, and climate-aware. The what-if tool, walking viewshed, and climate-aware routing stand on their own.

---

## 11. Scope

### Tier 1: Immediate scope (absolute deliverables)
1. Melbourne data pipeline: counts (with direction), merge-only ingest of the rolling live table, sensor locations, outage masking, sensor active periods, OSM graph, hand-checked sensor-to-edge matching; DVC versioning.
2. Feature pipeline: network metrics, slope, shade, surface/greenery, amenities with opening hours, weather, time (holidays, academic terms), events, transit access; visibility as a precomputed per-segment feature. Marks each feature as portable or Melbourne-only.
3. Flow model: gradient boosting baseline-ladder comparison, location-based splits, event-day evaluation, calibration curve; transferable features only; Melbourne-rich and portable variants (section 5).
4. Transfer evaluation (section 7.1): within-Melbourne precinct holdout (University of Melbourne edge, Carlton, North Melbourne; calibrate on 1–2 sensors) and a cross-city test on NYC DOT's bi-annual counts with the portable model and a full calibration curve. Moved up from Tier 2 because it's the only evidence behind any claim beyond Melbourne.
5. Aerial path segmentation at NEU campus scale, feeding detected paths into the graph (fallback: OSM paths only).
6. Agents: at least one, type TBD (LLM intake/planning assistant and/or walker sampling), with its evaluation set.
7. Event days: Melbourne event calendar (AFL fixtures and crowds, hand-curated major events), event features, separate evaluation centred on Marvel Stadium.
8. Web app: map, predicted flows, one or more what-if interactions; NEU Boston campus demonstration.
9. Full MLOps loop: tracking, registry, validation gates, CI/CD, Cloud Run deployment, monitoring against new monthly counts, scheduled and triggered retraining, expo status panel.
10. Data cards, license documentation, model cards.

### Tier 2: Stretch goals (if time and budget allow, roughly in priority order)
1. Sequential "what you see as you walk" viewshed visualization along routes.
2. Second agent type (whichever wasn't chosen in Tier 1).
3. Document upload for intake (event plans, traffic management PDFs).
4. Per-site calibration upload flow in the UI (customer counts, ticket scans).
5. Fenway game-day case study (MBTA surge + attendance check).
6. Synthetic parameter-recovery test.
7. Global-tier support: degraded-resolution training and the measured accuracy gap.
8. NEU Oakland campus transfer demonstration.
9. Fall Fest booth-exposure analysis (needs CSI table map).
10. Advan visit-weighted amenity attraction (if NEU has Dewey access).
11. Strava Metro validation (if the application is accepted).
12. Street-level surface/shade classification from open street-level imagery (e.g., Mapillary; CC BY-SA, so share-alike caution).

### Tier 3: Out of scope (future work; do not start without a team decision)
- Inverse reinforcement learning for route preferences (the upgrade path once the GBM baseline exists).
- Social-force or full microsimulation.
- Venue interiors.
- Crowd-safety certification or guarantees of any kind.
- Real-time streaming predictions.
- Native mobile apps.
- Global production coverage.
- Formal causal modeling of amenity effects.
- Class-change surge modeling on campus (needs class-schedule data; named as a known gap this semester, section 7.1).
- Learned ice/snow effects (no training data with freezing conditions; see section 7.1 for the interim rule-or-omit choice).

### Explicitly rejected (never)
- Phone or individual-level tracking.
- Training a shipped model on non-commercial datasets (demo-only fine-tuning that is never shipped is allowed).
- LLM-computed or LLM-estimated numbers, routes, or locations.
- Manual counting of people by the team.

---

## 12. Business model

- **Tiers:** campus/site license (one site, a few seats); city license (whole network, scenario builder, API); research mode (free or discounted, open GIS exports).
- **Unit economics:** cost per scenario stays low by design (precomputation, cheap inference, LLM only narrates finished results), so no per-query cost can outgrow the subscription price.

---

## 13. Open decisions and verification to-dos

**Decisions:** which agents; graph and feature-table formats; orchestration tool; weather source (proposed: ERA5 via Open-Meteo, already pulled); expo date and timeline; ice handling for Boston (labeled hand-set rule vs. omit); exact boundaries of the within-Melbourne transfer precincts.

**Decided (success criteria, reviewed by the team, Oct 2026):** flow model ≥15% lower MAE on log counts than Space Syntax + calibration on held-out sensors; 80% intervals cover 75–85%; NYC rank correlation ≥0.6 uncalibrated, and error on log totals halved after calibrating on 5 locations; segmentation recall ≥0.7 on known informal paths; intake agent ≥90% field-level accuracy with zero grounding violations reaching the user; p95 latency ≤2 s for precomputed scenarios. Full list in the scoping document, section 11.

**Verify:** Metro Tunnel station entrance locations and opening dates (for the graph); newest NEU buildings' heights in the Boston buildings data; MBTA gated entries; NEU Dewey access; Cloud Composer pricing; competitor feature claims (do microsimulation tools support outdoor/climate features?) before stating them in a pitch.

**Optional:** ask the CoM Open Data team (`opendata@melbourne.vic.gov.au`) for counts from Dec 2022 – Oct 2024, which no public source covers.

**Deadlines:** Strava Metro academic application closes **October 23, 2026**.
