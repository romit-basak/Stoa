# Stoa

Predicts where people walk, how many, and when, on every path of a site, and lets planners test changes ("close this path", "add an entrance", "it's game day") before building them.

IE7374 MLOps, Northeastern University, Fall 2026. *Formerly DesireLine.*

**Team:** Romit Basak · Arshmehar Kaur · Bimbasri Paduru · Puneet Singh Puri · Shubham Madhusudan Hadawle

## Case studies

| Site | Role |
|---|---|
| Melbourne | Training and validation. All accuracy claims rest here (City of Melbourne pedestrian sensors, 2009–present). |
| Melbourne precincts (UniMelb edge, Carlton, North Melbourne) | Within-city transfer test: held out whole |
| New York City | Cross-city transfer test on NYC DOT's open counts (evaluation only, never training) |
| NEU Boston campus | Demonstration (no local ground truth) |
| Event scenario | Demonstration. Melbourne event days (centred on Marvel Stadium) are evaluated separately. |

## Status

| Module | Path | Status |
|---|---|---|
| Ingestion | `src/ingest/` | ✅ Melbourne: counts, City of Melbourne layers, OSM, DEM, WorldCover, weather, AFL fixtures and crowds, calendars. ✅ NEU Boston: terrain, buildings, imagery, OSM, land cover, weather. ✅ NYC: test counts, OSM, land cover, weather |
| Features | `src/features/` | Planned |
| Flow model | `src/models/flow/` | Planned (baseline ladder → gradient boosting → learned models) |
| Path segmentation | `src/models/segmentation/` | Planned (NEU campus) |
| Agents | `src/agents/` | Planned; which agents is still TBD |
| Viewshed | `src/viewshed/` | Planned (integrating the existing engine) |
| API / web app | `src/api/`, `app/` | Planned |
| MLOps (DVC, MLflow, CI/CD, Cloud Run) | — | Planned |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Usage

### Pull data

Every source is configured in `configs/data.yaml`. Each script writes to `data/raw/<source>/` with a `MANIFEST.json` (URL, SHA-256, retrieval time, license).

```bash
# Melbourne (training)
.venv/bin/python -m src.ingest.melbourne_portal   # counts + archive, network, buildings, trees, POIs (~570 MB)
.venv/bin/python -m src.ingest.osm                # BBBike Melbourne extract (~90 MB)
.venv/bin/python -m src.ingest.dem                # GA 5 m LiDAR DEM clip + 30 m comparisons
.venv/bin/python -m src.ingest.worldcover         # ESA WorldCover 2021, clipped
.venv/bin/python -m src.ingest.weather            # hourly ERA5 via Open-Meteo, 2009–now
.venv/bin/python -m src.ingest.events             # AFL fixtures (Squiggle)
.venv/bin/python -m src.ingest.afl_attendance     # AFL crowd figures (AFL Tables)
.venv/bin/python -m src.ingest.calendars          # public holidays per site + Vic Important Dates

# NEU Boston (demonstration)
.venv/bin/python -m src.ingest.dem --site neu_boston          # USGS 3DEP 1 m clip
.venv/bin/python -m src.ingest.boston_buildings               # footprints with heights
.venv/bin/python -m src.ingest.imagery                        # NAIP 2023 clip + MassGIS 2025 tiles (~310 MB)

# NYC (cross-city transfer test only)
.venv/bin/python -m src.ingest.nyc                            # NYC DOT counts (~150 MB)

# OSM, land cover and weather for the other sites
.venv/bin/python -m src.ingest.osm --site neu_boston          # likewise worldcover, weather; and --site nyc
```

Options:
- `melbourne_portal --only <dataset-id> ...` fetches specific datasets.
- `melbourne_portal --large` also fetches the multi-GB 2018 DSM and point cloud.
- The live counts table is a rolling ~2-year window, so re-pulls are merged into the rows already held, never overwritten.
- `osm`, `dem`, `worldcover` and `weather` take `--site` (`melbourne`, `neu_boston`, `nyc`; see `configs/data.yaml`).
- School terms and university teaching periods are hand-maintained in `configs/calendars/`.

### Tests and lint

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
```

## Repository layout

```
configs/data.yaml        data sources and site bounding boxes
configs/calendars/       hand-maintained school terms and university teaching periods
data/                    raw/ and processed/ (git-ignored; versioned with DVC, planned)
docs/
  data_cards/            one card per dataset: schema, coverage, known issues
  data_licenses.md       license register; check before adding any dataset
  PROJECT_CONTEXT.md     reasoning behind decisions, scope tiers, validation plan
  scoping/               project scoping document (LaTeX source; the PDF is built, not tracked)
src/ingest/              ingestion scripts (common.py = download + manifest helpers)
tests/                   mirrors src/
CLAUDE.md                project rules and conventions
```

## Data at a glance

| Source | Coverage | License |
|---|---|---|
| Melbourne pedestrian counts | 2009-05 → 2022-10 (archive) and 2024-10 → now (live, rolling window); **no data Nov 2022 – Oct 2024**. Directional counts only in the live window. Metro Tunnel break from 2025-11-30. | CC BY 4.0 (portal-wide terms) |
| City of Melbourne layers (pedestrian network, footpath slope, buildings, trees, businesses, toilets, fountains, …) | Municipality | CC BY 4.0 |
| OpenStreetMap | Melbourne metro | ODbL 1.0 |
| GA 5 m LiDAR DEM | Study area (has land voids stored as 0) | CC BY 4.0 |
| ESA WorldCover 2021 | Study area | CC BY 4.0 |
| Weather (ERA5 via Open-Meteo) | 2009 → now, hourly | CC BY 4.0; free API is non-commercial |
| AFL games (Squiggle) + crowds (AFL Tables) | 2009 → now | No published licence (facts; credit both) |
| Holidays and calendars | 2007–2027 holidays; Vic school terms 2009–2026; UniMelb 2015–2026; NEU 2026–27 | `holidays` (MIT); Vic Gov CC BY 4.0; facts |
| NEU Boston: USGS 3DEP 1 m, Boston buildings, NAIP 2023, MassGIS 2025 orthos | Campus + ~400 m | Public domain; buildings PDDL |
| NYC DOT pedestrian counts | 114 locations, 2007 → May 2026, period totals | NYC Open Data Law (no use restrictions). **Evaluation only** |
| Stanford Drone Dataset (not pulled) | 8 scenes, 60 videos | CC BY-NC-SA 3.0: validation and demo-only fine-tuning, never shipped |

Details and caveats are in [`docs/data_cards/`](docs/data_cards/) and [`docs/data_licenses.md`](docs/data_licenses.md).

**Attribution:** Source: City of Melbourne Open Data · © OpenStreetMap contributors · © Commonwealth of Australia (Geoscience Australia) · © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) · Weather data by Open-Meteo.com (ERA5, Copernicus/ECMWF) · AFL data via Squiggle and AFL Tables · © State of Victoria · USGS 3DEP · USDA FSA NAIP · MassGIS (Bureau of Geographic Information), Commonwealth of Massachusetts EOTSS · City of Boston · NYC Department of Transportation, via NYC Open Data.

## Ground rules

The full list is in [`CLAUDE.md`](CLAUDE.md). The short version:

- Numbers come from models; the LLM only parses input and narrates results.
- Split by location, not by time. Never evaluate on sensors, areas or tiles seen in training.
- No personal data, no phone tracking, no manual people-counting.
- Non-commercial datasets are for validation (and demo-only fine-tuning), never in a shipped model.
- Features must transfer: no location IDs, no calendar month. Effects not learnable from Melbourne (ice, class-change surges) are labeled assumptions or left out.
- Outputs are planning guidance, not crowd-safety certification. Amenity what-ifs are labeled as rough guidance.

## Contributing

- Keep PRs small and focused. State which module and interface each PR touches.
- New features need tests. Model changes need an eval run.
- Config goes in `configs/`. No hard-coded paths, credentials or API keys.
- Before starting work, check its tier (Tier 1 deliverables before Tier 2 stretch goals). See `docs/PROJECT_CONTEXT.md` §11.
