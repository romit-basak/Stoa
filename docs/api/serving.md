# The prediction API

`src/api/` serves the registered flow model over HTTP. A caller sends street segments and local hours. It gets back the predicted pedestrians per hour with an 80% range, in the Predictions interface from CLAUDE.md: `(segment_id, hour, predicted_count, lower, upper, model_version)`. This first version serves the training site (Melbourne) with both variants, and runs locally. The Docker image, the Cloud Run deployment and publishing the champion come next.

## Endpoints

| | |
|---|---|
| `GET /health` | 200 with the loaded variants once a model is loaded, 503 before. For Cloud Run and smoke tests |
| `GET /model` | Which registered version serves each variant (`stoa-flow-rich v2`), its feature count and training metrics |
| `POST /predict` | Predictions for every segment at every hour |
| `GET /docs` | Interactive documentation (FastAPI) |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "variant": "rich",
  "segment_ids": ["715181647-2"],
  "hours": ["2026-11-23T08:00", "2026-11-23T13:00"]
}'
```

```json
{
  "site": "melbourne",
  "variant": "rich",
  "model_version": "stoa-flow-rich v2",
  "predictions": [
    {"segment_id": "715181647-2", "hour": "2026-11-23T08:00", "predicted_count": 1068.5, "lower": 344.6, "upper": 3238.3, "model_version": "stoa-flow-rich v2"},
    {"segment_id": "715181647-2", "hour": "2026-11-23T13:00", "predicted_count": 3722.0, "lower": 1207.6, "upper": 12134.9, "model_version": "stoa-flow-rich v2"}
  ],
  "sources": {
    "weather": "Open-Meteo forecast (CC BY 4.0)",
    "calendar_years_not_covered": [],
    "steepness": {"715181647-2": "sensor"},
    "segment_features": "OpenStreetMap (ODbL) and City of Melbourne open data"
  },
  "note": "Planning guidance: predicted passersby per hour with an 80% range, not a measured count."
}
```

- `hours` are local clock times at the site, without a UTC offset. The spring-forward hour, which doesn't exist on the clock, is refused. The repeated autumn hour is taken as the first (daylight-saving) one, as in training.
- `variant`: `rich` uses Melbourne's own layers. `portable` uses only features any city has; it is the variant meant for other cities.
- Limits: 500 segments, 168 hours (a week) and 20,000 segment-hours per request.
- Errors: an unknown `segment_id` returns 404 with the unknown IDs. An hour outside the weather forecast, a non-existent clock hour, or a request over the limits returns 422. A variant that isn't loaded returns 503.
- `sources` says where the inputs came from (core rule 3). `steepness` gives each segment's method: `sensor` (a count sensor's own values), `footpath_steepness` (the nearest survey section) or `ga_dem_5m` (terrain from the DEM).
- Predictions are planning guidance (core rule 7). On quiet streets the uncalibrated model can overshoot several times over: it predicted 357 against an actual 40 at a quiet Flinders Street crossing. Busy streets are much closer: at Town Hall it predicted 3,722 against an average of 3,272. That is the error pattern in the README's "Bias in the model".

## How a prediction's inputs are built

The model needs 163 features (rich) or 152 (portable), but a request names only a segment and an hour. `src/api/features.py` builds the rest **with the training code itself**, so served and trained features can't drift apart:

| Features | Source when serving |
|---|---|
| 134 segment features (network, amenities, land use, built form) | `osm_segment_features.parquet`, the table training's per-sensor features are rows of |
| Calendar, sun elevation, COVID and Metro Tunnel flags | `counts_hourly.time_features`, with the same holiday CSV and calendar YAMLs |
| 9 weather variables | Open-Meteo's forecast, joined by `counts_hourly.weather_for_counts` |
| Shade, events | `training_frame.shade_columns` and `event_columns` |
| Slope, local mean slope, elevation (rich) | `segment_steepness.parquet` (below) |
| Metro Tunnel stations' distance before they opened | The cap from `training_frame.design_matrix` |

`tests/api/test_parity.py` samples 3,000 real training rows and builds their features twice: through training's own path (`eligible`, `add_derived`, `Frame.matrix`) and through the API's, from only each row's segment and hour. Every column of both variants must match. Weather comes from the same archive on both sides, so the test isolates the feature code. It was checked by breaking the code on purpose: a one-hour weather shift, a dropped Metro Tunnel cap and a zeroed street feature each made it fail. It needs the DVC data, so it skips in CI and runs locally.

Loading a model also checks that it expects exactly the columns the API builds for its variant, so a schema change in training stops the API from starting instead of feeding a model the wrong columns.

### Slope and elevation for every segment

Training reads slope and elevation per count sensor (`src/features/sensor_steepness.py`), but a prediction can be for any of the 111,465 segments. `src/features/segment_steepness.py` (DVC stage and DAG task `segment_steepness`) gives every segment the same values, the same way, at its midpoint:
- **112 segments** with a count sensor on them take that sensor's values, so served features equal trained ones there;
- **40,520** take the nearest footpath-steepness section within 25 m, and the length-weighted mean within 25 m;
- **70,833** take elevation and terrain slope from the GA 5 m DEM.

DEM slope is terrain steepness, not footpath grade, as for the few sensors that use it. Most segments rely on it, so the rich model's slope features are less exact away from the surveyed footpaths.

### Weather: the one known difference

Training uses Open-Meteo's ERA5 archive, a reanalysis that lags real time by days. Serving uses Open-Meteo's forecast: the same provider and variables, but a forecast. The API asks for the past 92 and next 16 days, in GMT so the timestamps are UTC, and keeps the table for an hour. Hours outside that window are refused with the range in the message. The forecast is the API's only outside call per hour; everything else is loaded at startup.

## Running it locally

```bash
uv pip install --python .venv/bin/python -e ".[api]"
ops/vm/vm.sh tunnel        # in another terminal: MLflow at localhost:5001
MLFLOW_TRACKING_URI=http://localhost:5001 \
GOOGLE_APPLICATION_CREDENTIALS=$PWD/.secrets/stoa-airflow-dvc.json \
  .venv/bin/uvicorn src.api.app:app --port 8000
open http://localhost:8000/docs
```

It needs the DVC data (`dvc pull`) and `segment_steepness.parquet` (`python -m src.features.segment_steepness`, or the pipeline). The bucket key is needed because the model files are in `gs://stoa-dvc-data/mlflow-artifacts`.

| Setting | Default |
|---|---|
| `STOA_MODEL_URI_RICH`, `STOA_MODEL_URI_PORTABLE` | `models:/stoa-flow-<variant>@champion`. Any MLflow model URI, or a local folder with a saved bundle. Set one to `""` to skip that variant |
| `STOA_SITE` | the training site (`counts_hourly.site`) |

Only bundles can be served (`src/models/flow/bundle.py`: the mean model plus the 80% interval). The v1 champions are the mean model alone, so until v2 or later is champion, point the URIs at `@challenger`.

On 21 November 2026, with v2 of both variants loaded, 96 segment-hours took 0.43 s.

## Next

- A Docker image with the model and tables baked in, deployed to Cloud Run by a GitHub workflow, with a smoke test and rollback by traffic split.
- `publish_champion` in `stoa_train`: export the champion so a promotion triggers a deploy.
- Other sites (portable model and that site's tables), monitoring and drift checks.
