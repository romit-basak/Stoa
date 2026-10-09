# CLAUDE.md

Project context for Claude Code. Read this before making changes. Keep it updated as decisions are made; sections marked **TBD** are not yet decided by the team.

## What this project is

**Stoa** (formerly DesireLine) is a pedestrian planning tool for IE7374 (MLOps, Northeastern, Fall 2026). It predicts where people walk, how many, and when, across the paths and streets of a site, and lets planners test changes ("close this path", "add an entrance", "it's game day") before making them.

Users: city planners (primary), campus planners, event organizers (outdoor approach streets around venues only, never venue interiors), store owners choosing a site (hourly pass-by at a candidate storefront), and researchers (a generic research mode with standard GIS exports).

Case studies:
- **Melbourne**: training and validation. All accuracy claims rest here.
- **NEU Boston campus**: demonstration (no local ground truth). Not validated; any claim beyond Melbourne rests on the transfer tests (see "Generalization").
- **Event scenario** (stadium or campus event): demonstration.
- **Transfer tests:** held-out Melbourne precincts (within-city) and NYC DOT counts (cross-city). See "Generalization".
- **Marathons (Tier 2):** Melbourne Marathon (validation) and Boston Marathon (demonstration). See `docs/PROJECT_CONTEXT.md` 7.2.

## Core rules (do not violate)

1. **Numbers come from models, words come from the LLM.** The LLM may parse user input and narrate model outputs. It must never compute, estimate, or invent a count, a route, a location, or a claim about the network. Every number or network claim in agent output must trace to a model output or graph query, or the response is blocked.
2. **Split by location, not by time.** Never evaluate on sensors, areas, or image tiles that appear in training. Event days are evaluated separately.
3. **No invented inputs.** When parsing user input, missing values are asked for or filled with a visibly labeled default. Every field records its source: user text, default, or public data.
4. **Never guess a location silently.** Ambiguous places go to the map for the user to confirm.
5. **Licenses matter.** Non-commercial datasets (e.g., Stanford Drone, CC BY-NC-SA 3.0) are for validation and academic demonstration only: no model fine-tuned on them appears in shipped or distributed artifacts. A mirror can't relicense data: the original licence applies. Check `docs/data_licenses.md` before adding any dataset.
6. **No personal data.** No phone tracking, no individual-level data. Customer-uploaded data stays scoped to that customer's site.
7. **Planning guidance, not safety certification.** UI text and agent output must never present forecasts or crowd simulations as crowd-safety guarantees. Microsimulation invites safety readings, so these wording rules matter most there. Microsimulation covers outdoor approach areas only, never venue interiors.
8. **Amenity what-ifs are rough guidance.** Scenarios that add or remove shops, cafés, or other amenities must be labeled as such (see "Destinations and amenities"). Exposure scores (storefronts, booths, signs) are **visible impressions, not predicted customers**: our counts measure passersby, not store visits.
9. **Features must transfer.** No sensor IDs, location IDs, or raw coordinates as features. No calendar month or day of year: use physical quantities (temperature, sun elevation, daylight) instead, because Melbourne's seasons are reversed from Boston's.
10. **Don't present what wasn't learned as learned.** Melbourne never freezes, so any ice effect is a hand-set rule labeled as an assumption (or omitted). Campus predictions don't capture class-change surges; say so. Never claim the model generalizes globally.
11. **User knowledge and toggles are labeled scenario edits.** The learned model always gives the base prediction. Local knowledge becomes a structured adjustment whose strength the user picks from fixed levels (slight/moderate/strong); the LLM never sets the value. A toggle like "no shops here" edits the scenario's features (coherently, with caveats); it never swaps in a model trained without that feature.

## Architecture

```
Public data ──> ingest ──> validate ──> features ──> models ──> API ──> web app
                                                       │
                              agents (LLM intake; route-sampling walkers; microsimulation for limited areas)
                              viewshed engine (visibility features + walk views)
```

### Modules

| Module | Path | Purpose | Fallback if it underperforms |
|---|---|---|---|
| Ingestion | `src/ingest/` | Pull and version counts (Melbourne; NYC for testing), OSM, terrain, imagery, buildings, weather, events, calendars, POIs | Cached last-good snapshot |
| Features | `src/features/` | Network metrics (Space Syntax-style), slope, shade, POIs + opening hours, weather, time, event proximity | — |
| Flow model | `src/models/flow/` | Predict hourly pedestrian volume per segment | Space Syntax metrics + calibration |
| Path segmentation | `src/models/segmentation/` | Detect informal paths in aerial imagery, add to graph | OSM paths only |
| Agents | `src/agents/` | LLM intake (natural language -> event spec and knowledge adjustments); planning assistant is Tier 2 | Form-based input; static flow maps |
| Simulation | `src/simulation/` | Route-sampling walkers (first), then microsimulation for limited areas on an open-source simulator | Walkers only |
| Viewshed | `src/viewshed/` | 3D ray-casting visibility per segment and along routes | Precomputed visibility feature only |
| API | `src/api/` | Prediction and scenario endpoints | — |
| App | `app/` | Map, scenario builder, store-siting view with storefront exposure score, results; cached replay mode for demos | Recorded replay |

### Interfaces

Agree and freeze these early; build against fake data until real modules exist.
- **Graph:** nodes = intersections and entrances; edges = path segments with stable IDs. OSM graph built (GeoParquet, `segment_id` = `<way_id>-<k>`, stable within one pinned extract; see `docs/features/osm_features.md`). CoM-network graph and the shared format **TBD**.
- **Feature table:** one row per (segment_id, hour). Schema in `src/features/schema.py` **TBD**.
- **Predictions:** (segment_id, hour, predicted_count, lower, upper, model_version).
- **Event spec:** structured output of the intake agent. Schema in `src/agents/event_spec.py` **TBD**. Every field carries a `source` attribute.
- **Adjustment spec:** a parsed local-knowledge statement: segments, conditions, direction, user-chosen strength level, source text. Schema **TBD**.

## Data

See `docs/data_cards/` for one card per dataset and `docs/data_licenses.md` for terms.

| Dataset | Role | License notes |
|---|---|---|
| City of Melbourne Pedestrian Counting System | Train + validate | CC BY 4.0 (portal-wide terms; dataset field empty). Live table is a **rolling ~2-year window**: ingest merges, never overwrites |
| OpenStreetMap | Network, paths, amenities, opening hours | ODbL: share-alike applies to distributed derived databases |
| USGS 3DEP | Terrain (US) | Public domain |
| NAIP / MassGIS imagery | Path segmentation | Public domain; credit MassGIS |
| Sentinel-2, ESA WorldCover, Dynamic World | Surface, vegetation | Free/open; attribution required for CC BY layers |
| City of Boston buildings | NEU building heights | PDDL |
| Weather (all sites) | Hourly features | ERA5 via Open-Meteo (proposed default, pulled). CC BY 4.0; free API non-commercial |
| Events | Event features | AFL fixtures (Squiggle) and crowds (AFL Tables): facts, credit the source. Other events hand-curated with a source per row |
| Calendars | Holidays, school terms, university terms | `holidays` package (MIT); `configs/calendars/` (hand-maintained; never scrape unimelb.edu.au) |
| Stanford Drone Dataset | Route-choice validation; academic calibration demo | CC BY-NC-SA 3.0: validation and demonstration only; no model fine-tuned on it is shipped or distributed |
| Controlled pedestrian experiments (e.g., Jülich archive) | Microsimulation validation | **VERIFY** availability and licence |
| NYC DOT pedestrian counts | Cross-city transfer test only | NYC Open Data Law: no use restrictions. **Never train on it** (it would void the test) |

Data is versioned with DVC. Do not commit raw data to git.

### Destinations and amenities

People walk toward things, not just along pleasant routes. Amenities are attributes on the graph, not changes to its structure.

- **Sources:** OSM (shops, restaurants/cafés, public toilets, drinking water, benches, `opening_hours`); Melbourne open data business, café, toilet and fountain datasets (pulled; no opening hours); Advan visit counts via Dewey if NEU has access (visit-weighted attraction; US/Canada; academic terms, so validation/research only, never product training data).
- **Segment features:** counts of each amenity type within short *network* walking distances (not straight-line); an "open now" variant per hour from `opening_hours` (only ~12% of OSM amenities/shops carry it, so model it with explicit missingness); interactions with weather (fountains, shade, and toilets weighted more on hot days).
- **Walker simulation:** amenities act as destinations; simulated walkers pick destinations weighted by attraction and whether they're open.
- **Caveat (must be shown in the UI):** shops locate where foot traffic already is, so amenity features partly reflect traffic rather than cause it. Fine for predicting current flows; what-if scenarios that add or remove amenities ("add a café here") may overstate the effect and must be labeled as rough guidance.

## Evaluation

- **Baselines to beat:** shortest-path betweenness; Space Syntax metrics + calibration.
- **Flow model:** gradient boosting is the Tier 1 learned model (the GNN is Tier 2 and must beat it). Error on held-out sensors; improvement over baselines; calibration curve (accuracy vs. number of local sensors used); error by urban-density tier.
- **Microsimulation:** validated on controlled pedestrian-experiment data (speed–density), reported separately from the flow model. Melbourne counts validate flows, not crowd dynamics.
- **Knowledge adjustments:** enter the Metro Tunnel opening as a planner statement and check whether post-opening predictions improve.
- **Amenity what-ifs (Tier 2):** a before/after (difference-in-differences) test on large Melbourne openings near sensors. Feasible only at case-study scale: Emporium Melbourne (2014) is the one strong case.
- **Event days:** reported separately. AFL event days rest on **Marvel Stadium** (7 sensors within 500 m); the MCG has no sensor within 500 m, so MCG results are weak evidence.
- **Sanity scenarios** (must pass before deploy): e.g., closing a path lowers its predicted flow and raises flow on alternatives.
- **Segmentation:** accuracy on held-out tiles; recall of known paths.
- **Intake agent:** field-level extraction accuracy and location placement accuracy on `tests/agent_eval/`.
- **Transfer (Tier 1):** see "Generalization" below.

### Generalization

Full reasoning in `docs/PROJECT_CONTEXT.md` section 7.1.
- **Shape vs. scale:** the general model learns relative patterns; per-site calibration from local counts sets absolute volume. Calibration is the core of the generalization strategy, and the calibration curve is expected to be the strongest result.
- **Portable vs. Melbourne-rich model:** CoM-only layers (footpath-steepness survey, CLUE, business establishments, canopy polygons, CoM network) don't exist in NYC or Boston. Train a Melbourne-rich model for Melbourne accuracy and a **portable** model on features any city has, trained on an **OSM-derived Melbourne graph** so the NYC test measures city differences, not graph-source differences. The portable one is used for NYC and Boston. Report the gap.
- **Within-Melbourne transfer:** train on the CBD; hold out whole precincts of a different character: University of Melbourne edge (sensors 42-44 on Swanston St, campus-like but not interior campus paths), Carlton (Lygon St), North Melbourne. They have only 3, 5 and 6 sensors, so report uncalibrated and calibrated on 1–2 sensors; the full calibration curve runs on Melbourne overall and NYC.
- **Cross-city transfer:** train on Melbourne, test on NYC DOT's bi-annual counts (114 locations; period totals from two days a year, not hourly; compare with summed predicted hours). Evaluation only, never training. Report rank correlation separately from absolute error. Acceptance is "reported, not assumed"; rank correlation ≥ 0.6 is a goal, not pass/fail. Details in `docs/data_cards/nyc_dot_pedestrian_counts.md`.
- **Academic-term feature:** in-semester flag near universities from `configs/calendars/`. The UniMelb-edge sensors swing from ~0.45x their mean (Jan/Dec) to ~1.6x (March); CBD sensors don't. This covers semester rhythm, not class-change surges.
- **Structural break:** the Metro Tunnel opened 2025-11-30 (stations near Swanston St and the UniMelb edge). Split results at that date or include a station-access feature.
- **Claim to make:** learns patterns in Melbourne; the within-city test shows unfamiliar areas; the cross-city test shows a new city; a few local counts close much of the gap.

## MLOps lifecycle

- **Train:** experiments tracked in MLflow; every run records its DVC data version.
- **Package:** model + feature pipeline (same code as training) + viewshed engine in one Docker image; registered in MLflow with a model card.
- **Validate:** gates above; a model must beat the baselines and the currently deployed model.
- **Deploy:** Cloud Run; canary rollout with automatic rollback. Per-site calibrated variants stored separately.
- **Demo:** cached replay mode (record real runs, replay them at the expo) so the demo never depends on live calls.
- **Monitor:** accuracy against each month's new Melbourne counts; input and prediction drift; new OSM path edits; agent correction rates, grounding violations, and cost.
- **Retrain:** monthly on new counts; triggered by accuracy drop or drift; per site on new customer counts. Retrained models re-run all gates.

## Infrastructure

- GCP: Cloud Run (API, app), Cloud Storage (data, artifacts), Artifact Registry (images).
- CI/CD: GitHub Actions (tests, build, deploy on merge to main).
- Orchestration: **TBD.** Budget is limited; Cloud Composer is likely too expensive. Prefer Cloud Scheduler + Cloud Run jobs, or Airflow on a small VM.
- Keep costs per scenario low: precompute heavy work (baseline flows, viewsheds); the LLM narrates finished results and is not called per map interaction.

## Conventions

- Python ≥ 3.11 (`pyproject.toml`; exact version **TBD**). Environments via `uv`. Format with `ruff format`; lint with `ruff`. Type hints on public functions.
- Tests in `tests/`, mirroring `src/`. New features need tests; model changes need an eval run.
- Config in YAML under `configs/`; no hard-coded paths, credentials, or API keys. Secrets via environment variables / Secret Manager.
- Small, focused PRs. Each PR states which module and interface it touches.
- Code dependencies must have permissive licences (MIT, BSD, Apache), because the API image may be distributed. cityseer (AGPL-3.0) was ruled out for network metrics for this reason; `src/features/network_metrics.py` implements them with numba and scipy.
- **Documentation is Markdown or LaTeX, tracked in git.** Binary documents (`.pdf`, `.docx`, `.pptx`, `.xlsx`) are never tracked: PDFs are build outputs of the LaTeX sources, and any reference copies stay local. New write-ups go in `.md`; formatted deliverables go in `.tex` under `docs/`.
- Scoping document: `docs/scoping/project_scoping.tex`. Build: `cd docs/scoping && latexmk -pdf -outdir=build project_scoping.tex && cp build/project_scoping.pdf .`
- Commands (fill in as they exist):
  - Setup: `uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"`
  - Run tests: `.venv/bin/python -m pytest`; lint: `.venv/bin/ruff check src tests && .venv/bin/ruff format src tests`
  - Pull data: `.venv/bin/python -m src.ingest.<source> [--site melbourne|neu_boston|nyc]` (full list in README)
  - Build OSM features: `.venv/bin/python -m src.features.osm [--site melbourne] [--only extract,graph,features,sensors]` (column dictionary and join recipe in `docs/features/osm_features.md`)
  - Run pipeline: **TBD**
  - Run app locally: **TBD**

## Scope guardrails

Scope is tiered; the full lists are in `docs/PROJECT_CONTEXT.md` section 11.
- **Tier 1 (deliverables):** Melbourne pipeline, features, gradient-boosted flow model with baseline ladder, transfer evaluation (within-Melbourne precincts + NYC cross-city test), crowd simulation (route-sampling walkers, then microsimulation for limited areas), campus path segmentation, LLM intake agent with knowledge adjustments, event days, web app with what-ifs and the store-siting view (with storefront exposure score) on NEU Boston, full MLOps loop, data/model cards.
- **Tier 2 (stretch):** only after Tier 1 is solid. Spatio-temporal GNN, LLM planning assistant, 3D walk views, marathon case studies, research mode and terrain-grid graph builder, pedestrian signage/billboard placement, new-store before/after study (case-study scale; see `docs/analysis/new_store_openings_feasibility.md`), document intake, calibration upload UI, Fenway, NEU Oakland demonstration, global-resolution tier, Fall Fest, and others.
- **Tier 3 (future work):** IRL, city-wide microsimulation, venue interiors, safety certification, real-time streaming, native apps. Do not start these without a team decision.
- **Never:** phone tracking, shipping or distributing models fine-tuned on non-commercial data, LLM-computed numbers, manual people-counting.

Before working on anything, check which tier it's in. Don't let Tier 2 work delay Tier 1.

For background on any decision (data quirks, resolution tiers, amenities, validation plan, onboarding flows, bottlenecks), read `docs/PROJECT_CONTEXT.md`.

Checkpoint around weeks 5-6: if the learned model does not beat the Space Syntax baseline, stop investing in the model and shift the pitch to the interactive, 3D, climate-aware version.

## Open decisions

- Open-source microsimulator (JuPedSim, Vadere, or other), after checking licences
- Graph and feature-table formats (the OSM graph uses GeoParquet keyed on `segment_id`; confirm or change)
- Locations for the 18 archive sensors missing from the current sensor-locations file (17.5% of 2009–2022 rows)
- Orchestration tool
- Weather data source (proposed: ERA5 via Open-Meteo, already pulled for all sites)
- Ice handling for Boston: labeled hand-set rule, or omit
