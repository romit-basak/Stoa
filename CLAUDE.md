# CLAUDE.md

Project context for Claude Code. Read this before making changes. Keep it updated as decisions are made; sections marked **TBD** are not yet decided by the team.

## What this project is

**Stoa** (formerly DesireLine) is a pedestrian planning tool for IE7374 (MLOps, Northeastern, Fall 2026). It predicts where people walk, how many, and when, across the paths and streets of a site, and lets planners test changes ("close this path", "add an entrance", "it's game day") before making them.

Users: city planners (primary), campus planners, event organizers (outdoor approach streets around venues only, never venue interiors), and researchers (a generic research mode with standard GIS exports).

Case studies:
- **Melbourne**: training and validation. All accuracy claims rest here.
- **NEU Boston campus**: demonstration (no local ground truth).
- **Event scenario** (stadium or campus event): demonstration.

## Core rules (do not violate)

1. **Numbers come from models, words come from the LLM.** The LLM may parse user input and narrate model outputs. It must never compute, estimate, or invent a count, a route, a location, or a claim about the network. Every number or network claim in agent output must trace to a model output or graph query, or the response is blocked.
2. **Split by location, not by time.** Never evaluate on sensors, areas, or image tiles that appear in training. Event days are evaluated separately.
3. **No invented inputs.** When parsing user input, missing values are asked for or filled with a visibly labeled default. Every field records its source: user text, default, or public data.
4. **Never guess a location silently.** Ambiguous places go to the map for the user to confirm.
5. **Licenses matter.** Non-commercial datasets (e.g., Stanford Drone) are for validation only and must never enter training data for the product model. Check `docs/data_licenses.md` before adding any dataset.
6. **No personal data.** No phone tracking, no individual-level data. Customer-uploaded data stays scoped to that customer's site.
7. **Planning guidance, not safety certification.** UI text and agent output must never present forecasts as crowd-safety guarantees.
8. **Amenity what-ifs are rough guidance.** Scenarios that add or remove shops, cafés, or other amenities must be labeled as such (see "Destinations and amenities").

## Architecture

```
Public data ──> ingest ──> validate ──> features ──> models ──> API ──> web app
                                                       │
                              agents (LLM intake / planning assistant, walkers)
                              viewshed engine (visibility features + walk views)
```

### Modules

| Module | Path | Purpose | Fallback if it underperforms |
|---|---|---|---|
| Ingestion | `src/ingest/` | Pull and version Melbourne counts, OSM, terrain, imagery, weather, events, POIs | Cached last-good snapshot |
| Features | `src/features/` | Network metrics (Space Syntax-style), slope, shade, POIs + opening hours, weather, time, event proximity | — |
| Flow model | `src/models/flow/` | Predict hourly pedestrian volume per segment | Space Syntax metrics + calibration |
| Path segmentation | `src/models/segmentation/` | Detect informal paths in aerial imagery, add to graph | OSM paths only |
| Agents | `src/agents/` | LLM intake (natural language -> event spec) and planning assistant; walker simulation **TBD** | Form-based input; static flow maps |
| Viewshed | `src/viewshed/` | 3D ray-casting visibility per segment and along routes | Precomputed visibility feature only |
| API | `src/api/` | Prediction and scenario endpoints | — |
| App | `app/` | Map, scenario builder, results | — |

### Interfaces

Agree and freeze these early; build against fake data until real modules exist.
- **Graph:** nodes = intersections and entrances; edges = path segments with stable IDs. Format **TBD**.
- **Feature table:** one row per (segment_id, hour). Schema in `src/features/schema.py` **TBD**.
- **Predictions:** (segment_id, hour, predicted_count, lower, upper, model_version).
- **Event spec:** structured output of the intake agent. Schema in `src/agents/event_spec.py` **TBD**. Every field carries a `source` attribute.

## Data

See `docs/data_cards/` for one card per dataset and `docs/data_licenses.md` for terms.

| Dataset | Role | License notes |
|---|---|---|
| City of Melbourne Pedestrian Counting System | Train + validate | **Verify license before use** |
| OpenStreetMap | Network, paths, amenities, opening hours | ODbL: share-alike applies to distributed derived databases |
| USGS 3DEP | Terrain (US) | Public domain |
| NAIP / MassGIS imagery | Path segmentation | NAIP public domain; **verify MassGIS** |
| Sentinel-2, ESA WorldCover, Dynamic World | Surface, vegetation | Free/open; attribution required for CC BY layers |
| Weather (Melbourne) | Hourly features | **TBD source** |
| Event calendars | Event features | Public schedules |
| Stanford Drone Dataset | Route-choice validation only | Non-commercial: **never train on it** |

Data is versioned with DVC. Do not commit raw data to git.

### Destinations and amenities

People walk toward things, not just along pleasant routes. Amenities are attributes on the graph, not changes to its structure.

- **Sources:** OSM (shops, restaurants/cafés, public toilets, drinking water, benches, `opening_hours`); Melbourne open data business/toilet/fountain datasets (**verify availability**); Advan visit counts via Dewey if NEU has access (visit-weighted attraction; US/Canada; academic terms, so validation/research only, never product training data).
- **Segment features:** counts of each amenity type within short *network* walking distances (not straight-line); an "open now" variant per hour from `opening_hours`; interactions with weather (fountains, shade, and toilets weighted more on hot days).
- **Walker simulation:** amenities act as destinations; simulated walkers pick destinations weighted by attraction and whether they're open.
- **Caveat (must be shown in the UI):** shops locate where foot traffic already is, so amenity features partly reflect traffic rather than cause it. Fine for predicting current flows; what-if scenarios that add or remove amenities ("add a café here") may overstate the effect and must be labeled as rough guidance.

## Evaluation

- **Baselines to beat:** shortest-path betweenness; Space Syntax metrics + calibration.
- **Flow model:** error on held-out sensors; improvement over baselines; event-day error reported separately; calibration curve (accuracy vs. number of local sensors used).
- **Sanity scenarios** (must pass before deploy): e.g., closing a path lowers its predicted flow and raises flow on alternatives.
- **Segmentation:** accuracy on held-out tiles; recall of known paths.
- **Intake agent:** field-level extraction accuracy and location placement accuracy on `tests/agent_eval/`.

## MLOps lifecycle

- **Train:** experiments tracked in MLflow; every run records its DVC data version.
- **Package:** model + feature pipeline (same code as training) + viewshed engine in one Docker image; registered in MLflow with a model card.
- **Validate:** gates above; a model must beat the baselines and the currently deployed model.
- **Deploy:** Cloud Run; canary rollout with automatic rollback. Per-site calibrated variants stored separately.
- **Monitor:** accuracy against each month's new Melbourne counts; input and prediction drift; new OSM path edits; agent correction rates, grounding violations, and cost.
- **Retrain:** monthly on new counts; triggered by accuracy drop or drift; per site on new customer counts. Retrained models re-run all gates.

## Infrastructure

- GCP: Cloud Run (API, app), Cloud Storage (data, artifacts), Artifact Registry (images).
- CI/CD: GitHub Actions (tests, build, deploy on merge to main).
- Orchestration: **TBD.** Budget is limited; Cloud Composer is likely too expensive. Prefer Cloud Scheduler + Cloud Run jobs, or Airflow on a small VM.
- Keep costs per scenario low: precompute heavy work (baseline flows, viewsheds); the LLM narrates finished results and is not called per map interaction.

## Conventions

- Python 3.11 **(TBD)**. Format with `ruff format`; lint with `ruff`. Type hints on public functions.
- Tests in `tests/`, mirroring `src/`. New features need tests; model changes need an eval run.
- Config in YAML under `configs/`; no hard-coded paths, credentials, or API keys. Secrets via environment variables / Secret Manager.
- Small, focused PRs. Each PR states which module and interface it touches.
- Commands (fill in as they exist):
  - Setup: **TBD**
  - Run tests: **TBD**
  - Run pipeline: **TBD**
  - Run app locally: **TBD**

## Scope guardrails

Scope is tiered; the full lists are in `docs/PROJECT_CONTEXT.md` section 11.
- **Tier 1 (deliverables):** Melbourne pipeline, features, flow model with baseline ladder, campus path segmentation, at least one agent, event days, web app with what-ifs on NEU Boston, full MLOps loop, data/model cards.
- **Tier 2 (stretch):** only after Tier 1 is solid. Walking-view visualization, second agent, document intake, calibration upload, Fenway, transfer tests, global-resolution tier, Fall Fest, and others.
- **Tier 3 (future work):** IRL, microsimulation, venue interiors, safety certification, real-time streaming, native apps. Do not start these without a team decision.
- **Never:** phone tracking, training on non-commercial data, LLM-computed numbers, manual people-counting.

Before working on anything, check which tier it's in. Don't let Tier 2 work delay Tier 1.

For background on any decision (data quirks, resolution tiers, amenities, validation plan, onboarding flows, bottlenecks), read `docs/PROJECT_CONTEXT.md`.

Checkpoint around weeks 5-6: if the learned model does not beat the Space Syntax baseline, stop investing in the model and shift the pitch to the interactive, 3D, climate-aware version.

## Open decisions

- Which agents: LLM intake/planning assistant, walker simulation, or both
- Graph and feature-table formats
- Orchestration tool
- Weather data source
- Melbourne and MassGIS license confirmation