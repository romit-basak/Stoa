# Stoa

Predicts where people walk, how many, and when, on every path of a site, and lets planners test changes ("close this path", "add an entrance", "it's game day") before building them.

IE7374 MLOps, Northeastern University, Fall 2026. *Formerly DesireLine.*

**Team:** Romit Basak · Arshmehar Kaur · Bimbasri Paduru · Puneet Singh Puri · Shubham Madhusudan Hadawle

## Case studies

| Site | Role |
|---|---|
| Melbourne | Training and validation. All accuracy claims rest here (City of Melbourne pedestrian sensors, 2009–present). |
| NEU Boston campus | Demonstration (no local ground truth) |
| Event scenario | Demonstration. Melbourne event days are evaluated separately. |

## Status

| Module | Path | Status |
|---|---|---|
| Ingestion | `src/ingest/` | ✅ Melbourne: counts, City of Melbourne layers, OSM, DEM, WorldCover, weather, AFL events |
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
.venv/bin/python -m src.ingest.melbourne_portal   # counts + archive, network, buildings, trees, POIs (~570 MB)
.venv/bin/python -m src.ingest.osm                # BBBike Melbourne extract (~90 MB)
.venv/bin/python -m src.ingest.dem                # GA 5 m LiDAR DEM clip + 30 m comparisons
.venv/bin/python -m src.ingest.worldcover         # ESA WorldCover 2021, clipped
.venv/bin/python -m src.ingest.weather            # hourly ERA5 via Open-Meteo, 2009–now
.venv/bin/python -m src.ingest.events             # AFL games (MCG, Marvel Stadium)
```

Options:
- `melbourne_portal --only <dataset-id> ...` fetches specific datasets.
- `melbourne_portal --large` also fetches the multi-GB 2018 DSM and point cloud.
- `osm`, `dem`, `worldcover` and `weather` take `--site` to select a site from `configs/data.yaml` (only `melbourne` is configured so far).

### Tests and lint

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
```

## Repository layout

```
configs/data.yaml        data sources and site bounding boxes
data/                    raw/ and processed/ (git-ignored; versioned with DVC, planned)
docs/
  data_cards/            one card per dataset: schema, coverage, known issues
  data_licenses.md       license register; check before adding any dataset
  PROJECT_CONTEXT.md     reasoning behind decisions, scope tiers, validation plan
src/ingest/              ingestion scripts (common.py = download + manifest helpers)
tests/                   mirrors src/
CLAUDE.md                project rules and conventions
```

## Data at a glance

| Source | Coverage | License |
|---|---|---|
| Melbourne pedestrian counts | 2009-05 → 2022-10 (archive) and 2024-10 → now (live); **no data Nov 2022 – Oct 2024**. Directional counts only in the live window. | **Unconfirmed**, verification pending |
| City of Melbourne layers (pedestrian network, footpath slope, buildings, trees, businesses, toilets, fountains, …) | Municipality | CC BY 4.0 |
| OpenStreetMap | Melbourne metro | ODbL 1.0 |
| GA 5 m LiDAR DEM | Study area (has land voids stored as 0) | CC BY 4.0 |
| ESA WorldCover 2021 | Study area | CC BY 4.0 |
| Weather (ERA5 via Open-Meteo) | 2009 → now, hourly | CC BY 4.0; free API is non-commercial |
| AFL games (Squiggle) | 2009 → now | No published licence |

Details and caveats are in [`docs/data_cards/`](docs/data_cards/) and [`docs/data_licenses.md`](docs/data_licenses.md).

**Attribution:** Source: City of Melbourne Open Data · © OpenStreetMap contributors · © Commonwealth of Australia (Geoscience Australia) · © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) · Weather data by Open-Meteo.com (ERA5, Copernicus/ECMWF) · AFL data via Squiggle.

## Ground rules

The full list is in [`CLAUDE.md`](CLAUDE.md). The short version:

- Numbers come from models; the LLM only parses input and narrates results.
- Split by location, not by time. Never evaluate on sensors, areas or tiles seen in training.
- No personal data, no phone tracking, no manual people-counting.
- Non-commercial datasets are for validation only, never for training.
- Outputs are planning guidance, not crowd-safety certification. Amenity what-ifs are labeled as rough guidance.

## Contributing

- Keep PRs small and focused. State which module and interface each PR touches.
- New features need tests. Model changes need an eval run.
- Config goes in `configs/`. No hard-coded paths, credentials or API keys.
- Before starting work, check its tier (Tier 1 deliverables before Tier 2 stretch goals). See `docs/PROJECT_CONTEXT.md` §11.
