# Stoa

Predicts where people walk, how many, and when, on every path of a site, and lets planners test changes ("close this path", "add an entrance", "it's game day") before building them. It simulates crowds in limited outdoor areas and shows store owners the hourly pass-by at a candidate storefront and how many passersby can see it.

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
| Marathons (stretch) | Melbourne Marathon (validation) and Boston Marathon (demonstration) |

## Status

| Module | Path | Status |
|---|---|---|
| Ingestion | `src/ingest/` | ✅ Melbourne: counts, City of Melbourne layers, OSM, DEM, WorldCover, weather, AFL fixtures and crowds, calendars. ✅ NEU Boston: terrain, buildings, imagery, OSM, land cover, weather. ✅ NYC: test counts, OSM, land cover, weather |
| Features | `src/features/` | ✅ OSM: walking graph, per-segment features (tags, amenities, opening hours, built form, transit, network and angular metrics), sensor matching for Melbourne. See [docs/features/osm_features.md](docs/features/osm_features.md). ✅ Footpath slope and elevation per sensor. ✅ Hourly table for Melbourne: counts + time + weather + calendars + slope ([docs/features/counts_hourly.md](docs/features/counts_hourly.md)). Shade and other City of Melbourne layers planned |
| Flow model | `src/models/flow/` | Planned (baseline ladder → gradient boosting; spatio-temporal GNN is a stretch goal) |
| Path segmentation | `src/models/segmentation/` | Planned (NEU campus) |
| Agents | `src/agents/` | Planned: LLM intake agent with local-knowledge adjustments (planning assistant is a stretch goal) |
| Simulation | `src/simulation/` | Planned: route-sampling walkers, then microsimulation for limited areas |
| Viewshed | `src/viewshed/` | Planned (integrating the existing engine) |
| API / web app | `src/api/`, `app/` | Planned (map, scenario builder, store-siting view, cached replay mode) |
| MLOps (Airflow, DVC, MLflow, CI/CD, Cloud Run) | `dags/` | ✅ Airflow: daily `stoa_pipeline` DAG with retries, failure emails and logging. ✅ DVC (data in `gs://stoa-dvc-data`). ✅ Bias slicing: representation report per slice ([docs/bias/bias_slicing.md](docs/bias/bias_slicing.md)); per-slice model errors next. Data validation in progress. MLflow, CI/CD, Cloud Run planned |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Without uv, `requirements.txt` pins the exact versions we tested with: `pip install -r requirements.txt` (Python 3.11). `pyproject.toml` is the source of truth; after changing its dependencies, regenerate the pins with `uv pip compile pyproject.toml --extra dev --python-version 3.11 -o requirements.txt`.

### Get the data (DVC)

Data is not in git. It is versioned with [DVC](https://dvc.org) and stored in the `gs://stoa-dvc-data` bucket on the team's GCP project (ask Romit for access). Git holds only the pointer files `data/raw.dvc`, `data/interim.dvc` and `data/processed.dvc`.

```bash
gcloud auth application-default login    # once; sign in with the account added to the project
.venv/bin/dvc pull                       # downloads data/ at the version this commit points to
```

The pipeline's `dvc_push` task saves and uploads a new version after every run that passes validation (see "Run the pipeline"). To do it by hand after changing data, save a new version and commit its pointers:

```bash
.venv/bin/dvc add data/raw data/interim data/processed
.venv/bin/dvc push
git add data/*.dvc && git commit -m "..."
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
- `osm` keeps an existing extract (graph segment IDs belong to one extract); `--refresh` downloads the latest.
- `weather` re-fetches only missing years and the last two; `--full` re-fetches every year.
- School terms and university teaching periods are hand-maintained in `configs/calendars/`.

### Build features

```bash
.venv/bin/python -m src.features.osm --site melbourne     # OSM graph + segment features + sensor matching (~2.5 min)
.venv/bin/python -m src.features.sensor_steepness         # footpath slope and elevation per sensor (~5 s)
.venv/bin/python -m src.features.counts_hourly            # hourly table: counts + time + weather + calendars + slope (~30 s)
.venv/bin/python -m src.bias.representation               # bias slices: how much data each group has (~35 s)
```

Outputs go to `data/interim/osm/<site>/` (GeoParquet layers and the graph) and `data/processed/features/<site>/`. `sensor_osm_features.parquet` joins onto the counts on `location_id`. The column dictionary, join recipe and caveats are in [docs/features/osm_features.md](docs/features/osm_features.md). `counts_hourly.parquet` is the hourly training table (one row per sensor and hour); see [docs/features/counts_hourly.md](docs/features/counts_hourly.md), including how it handles daylight saving. `sensor_steepness.parquet` gives each outdoor sensor its footpath slope and elevation (from Shubham Hadawle's join notebook), and the hourly table carries those columns.

### Run the pipeline (Airflow)

The Melbourne pipeline runs as one Airflow DAG, `stoa_pipeline` (`dags/stoa_pipeline.py`), in Docker. Requires Docker Desktop.

On Linux, first tell the containers to run as your user, or they can't write to `data/`. Docker Desktop on macOS and Windows handles this for you.

```bash
echo "AIRFLOW_UID=$(id -u)" >> .env
```

```bash
docker compose up -d --build     # first run builds the image (a few minutes)
# open http://localhost:8080 (no login locally), unpause stoa_pipeline, press Trigger
docker compose down              # stop; add -v to also wipe Airflow's database
```

The repo is mounted into the containers, so code changes apply on the next run; changes to `pyproject.toml` dependencies need `--build`. Data is written to `data/` on the host, as when running the modules directly.

```
pull_counts ────────────────────────────────────────────┐
pull_osm ─► osm_extract ─► osm_graph ─► osm_segment_features ─► osm_sensor_match ─┐
pull_static_layers ─► sensor_steepness ───────────────────────────────────────────┤
pull_weather ─────────────────────────────────────────────────────────────────────┼─► join_all ─┬─► validate ────┬─► dvc_push
pull_calendars ───────────────────────────────────────────────────────────────────┘             └─► bias_slices ─┘
```

(`pull_counts` also feeds `osm_sensor_match` and `sensor_steepness`, which need the sensor locations. `pull_static_layers` fetches the footpath-steepness survey, the 5 m DEM, the CLUE areas and the AFL crowds only when they aren't held yet, and also feeds `bias_slices`. `bias_slices` reports how much data each bias slice has; see [docs/bias/bias_slicing.md](docs/bias/bias_slicing.md).) It runs daily so no day of the counts' rolling window is missed; each task retries twice. `validate` is a placeholder: replace its body and keep its name, and the wiring stays the same. `dvc_push` saves the result as a new DVC version (below). A daily run takes about 30 seconds; `join_all` is the slowest task (see "Pipeline optimization" below).

What a daily run re-does:
- **Counts:** only the days from two before the latest date held, merged into the rows already held on (location, date, hour). The fixed 2009–2022 archive is kept. `python -m src.ingest.melbourne_portal --only pedestrian-counting-system-monthly-counts-per-hour --full` re-fetches everything.
- **Weather:** only the last two years (`--full` re-fetches all).
- **OSM:** the extract is pinned. `pull_osm` keeps the file it has, and each OSM stage skips itself unless one of its inputs changed: the extract, `configs/features.yaml`, the code in `src/features/`, the sensor locations, or `configs/sensor_overrides.yaml`. A skipped OSM chain takes a couple of seconds.

To move to a newer OSM extract, trigger the DAG with the `refresh_osm` parameter set to true (Trigger → advanced options in the UI). That renumbers the graph segments, so `osm_sensor_match` fails until any sensor overrides that point at old segments are fixed.

**Failure alerts.** When a task fails after its last retry, Airflow emails the team (task, run and a link to the log). Settings live in `.env` in the repo root, which is git-ignored; without it, alerts are off. The sender needs a Gmail [App Password](https://myaccount.google.com/apppasswords) (requires 2-Step Verification), not the account password:

```bash
STOA_SMTP_USER=you@gmail.com
STOA_SMTP_PASSWORD=abcdefghijklmnop          # 16-character App Password, no spaces
STOA_ALERT_EMAILS=a@northeastern.edu,b@northeastern.edu
```

After editing `.env`, run `docker compose up -d` to apply it.

**Data versioning (`dvc_push`).** The last task runs `python -m src.pipeline.dvc_push`: `dvc add` on `data/raw`, `data/interim` and `data/processed`, then `dvc push` to `gs://stoa-dvc-data`. It runs after `validate`, so only data that passed the checks becomes a version. It does not commit to git: when the data changed, its log ends with the `git add data/*.dvc && git commit` command to run. Commit those pointers to record the version; until then, teammates' `dvc pull` still fetches the previous one.

The containers log in to the bucket as the `stoa-airflow-dvc` service account, through a key file in `.secrets/` (git-ignored, never shared; each person makes their own). Once, with `gcloud` signed in to an account on the project:

```bash
mkdir -p .secrets
gcloud iam service-accounts keys create .secrets/stoa-airflow-dvc.json \
  --iam-account=stoa-airflow-dvc@stoa-511822.iam.gserviceaccount.com
```

`dvc_push` finds credentials the way Google's own tools do: this key file, else your `gcloud auth application-default login`, else the service account of the Google Cloud VM it runs on (so a VM needs no key file).

With no credentials at all, every other task still runs and `dvc_push` is **skipped** (shown pink in the UI) with a warning in its log, so the run still succeeds and no failure email is sent. That is the expected result on a clone by someone outside the team: the pipeline builds all the data from public sources, it just isn't versioned. On team machines, add this to `.env` so a missing key fails the task and sends the failure email instead:

```bash
STOA_REQUIRE_DVC_PUSH=1
```

**Logs and error handling.** Every command logs what it does with Python's `logging` (set up once in `src/ingest/common.py`): each download with its size, retries, skipped steps and row counts. In Airflow, open a run, click a task and choose Logs. The same files are written to `logs/` in the repo (mounted in `docker-compose.yaml`, git-ignored), one folder per DAG, run and task, e.g. `logs/dag_id=stoa_pipeline/run_id=<run>/task_id=pull_counts/attempt=1.log`, so you can read or `grep` them without the UI. `download()` retries rate limits, server errors, dropped connections, timeouts and empty or cut-short files with backoff, writing to a `.part` file so a failed download never replaces the last good copy. If every try fails it raises `DownloadError` naming the URL; the task then fails, Airflow retries it twice, and the failure email links to the log.

### Pipeline optimization

Airflow's Gantt chart (a run → Gantt) showed one bottleneck: `pull_counts` took 229 of 234 seconds, and every task after the OSM chain waited for it, because it re-downloaded the whole 2-year counts table (1.6M rows) and the 42 MB 2009–2022 archive on every run, to add one new day.

Fix: `pull_counts` now asks the portal only for rows from two days before the latest date held (the portal fills in the current day hour by hour, and older rows don't change), and keeps held attachments.

| Task | Before (s) | After (s) |
|---|---|---|
| `pull_counts` | 229.3 | 5.0 |
| Other ingest tasks | 7.1 | 7.2 |
| OSM stages (skipped, inputs unchanged) | 2.9 | 2.6 |
| `join_all` | placeholder | 22.5 |
| **Whole run** | **233.6** | **31.5** |

| Before (run of 2026-10-17 00:00 UTC) | After (run `after_incremental_counts_2`) |
|---|---|
| ![Gantt before](docs/pipeline/gantt_before.png) | ![Gantt after](docs/pipeline/gantt_after.png) |

`join_all` is now the slowest task. A run where `src/features/` or the OSM inputs changed also rebuilds the OSM stages (about 2.5 minutes, mostly `osm_segment_features`).

While doing this we found that the portal's `id` is not unique (it glues location, hour and date: location 4 at hour 20 and location 42 at hour 2 share an id), so merging on it had dropped about 4.5% of rows. Counts are now merged on (location, date, hour).

### Tests and lint

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
```

The DAG tests (`tests/dags/`) need Airflow, so they skip in `.venv`. Run them in the Airflow image (no Postgres needed; the override gives the run a throwaway database):

```bash
docker compose run --rm --no-deps \
  -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////tmp/airflow.db airflow-init \
  bash -c "pip install -q pytest && cd /opt/stoa && python -m pytest tests/dags"
```

## Repository layout

```
configs/data.yaml        data sources and site bounding boxes
configs/calendars/       hand-maintained school terms and university teaching periods
configs/analysis.yaml    parameters for src/analysis/
configs/features.yaml    feature pipeline settings (walk filter, POI categories, radii, sensor matching)
configs/sensor_overrides.yaml  hand-checked sensor-to-segment matches
configs/bias.yaml        bias slices: precincts, time and weather bins, eras, thresholds
dags/                    Airflow DAGs (stoa_pipeline.py); Dockerfile + docker-compose.yaml run them
data/                    raw/, interim/ and processed/ (git-ignored; versioned with DVC, pointers in data/*.dvc)
logs/                    Airflow task logs (written by the containers, git-ignored)
docs/
  data_cards/            one card per dataset: schema, coverage, known issues
  analysis/              write-ups of one-off analyses
  features/              feature tables: how to build, column dictionaries, joins
  bias/                  bias analysis: slices, representation report, findings
  data_licenses.md       license register; check before adding any dataset
  PROJECT_CONTEXT.md     reasoning behind decisions, scope tiers, validation plan
  scoping/               project scoping document (LaTeX source; the PDF is built, not tracked)
src/ingest/              ingestion scripts (common.py = download + manifest helpers)
src/features/            feature pipeline (osm.py runs extract -> graph -> features -> sensors)
src/bias/                bias slicing (slices.py) and the representation report
src/analysis/            one-off analyses (e.g. new_openings: before/after study feasibility)
tests/                   mirrors src/; tests/dags/ checks the DAG (run in the Airflow image)
requirements.txt         pinned versions generated from pyproject.toml
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
| Stanford Drone Dataset (not pulled) | 8 scenes, 60 videos | CC BY-NC-SA 3.0: validation and academic demonstration only; no model fine-tuned on it is shipped or distributed |

Details and caveats are in [`docs/data_cards/`](docs/data_cards/) and [`docs/data_licenses.md`](docs/data_licenses.md).

**Attribution:** Source: City of Melbourne Open Data · © OpenStreetMap contributors · © Commonwealth of Australia (Geoscience Australia) · © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) · Weather data by Open-Meteo.com (ERA5, Copernicus/ECMWF) · AFL data via Squiggle and AFL Tables · © State of Victoria · USGS 3DEP · USDA FSA NAIP · MassGIS (Bureau of Geographic Information), Commonwealth of Massachusetts EOTSS · City of Boston · NYC Department of Transportation, via NYC Open Data.

## Ground rules

The full list is in [`CLAUDE.md`](CLAUDE.md). The short version:

- Numbers come from models; the LLM only parses input and narrates results.
- Split by location, not by time. Never evaluate on sensors, areas or tiles seen in training.
- No personal data, no phone tracking, no manual people-counting.
- Non-commercial datasets are for validation and academic demonstration only; no model fine-tuned on them is shipped or distributed.
- Features must transfer: no location IDs, no calendar month. Effects not learnable from Melbourne (ice, class-change surges) are labeled assumptions or left out.
- Outputs, including crowd simulations, are planning guidance, not crowd-safety certification. Amenity what-ifs and user-entered local knowledge are labeled as assumptions.

## Contributing

- Keep PRs small and focused. State which module and interface each PR touches.
- New features need tests. Model changes need an eval run.
- Config goes in `configs/`. No hard-coded paths, credentials or API keys.
- Before starting work, check its tier (Tier 1 deliverables before Tier 2 stretch goals). See `docs/PROJECT_CONTEXT.md` §11.
