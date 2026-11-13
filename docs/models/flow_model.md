# Flow model

The flow model predicts the hourly pedestrian count on a path segment. It is trained on the Melbourne counts and can predict any segment with features, so it covers the whole network, not only the sensors. Code is in `src/models/flow/`, settings in `configs/models.yaml` (`flow`), and every run is logged to MLflow as described in `experiment_tracking.md`.

## Models

| Model | What it uses | Role |
|---|---|---|
| `baseline_betweenness` | Shortest-path betweenness at 1600 m | Ladder rung 1 |
| `baseline_space_syntax` | Every Space Syntax-style metric: metric betweenness, closeness, reach, junctions, angular choice and integration, at 400, 800 and 1600 m | Ladder rung 2, and the bar the gate measures against |
| `hgb_<variant>` | scikit-learn `HistGradientBoostingRegressor`, Poisson loss | Candidate |
| `lgbm_<variant>` | LightGBM, Poisson objective | Candidate |
| `xgb_<variant>` | XGBoost, `count:poisson`, histogram trees | Candidate |

Space Syntax gives movement potential per segment, not hourly counts, and needs calibrating against counts. So both baselines take the same form: the typical log count for each hour of the week, plus a ridge regression of the remainder on the rung's network terms (log1p), both fitted on the training sensors. The network terms are constant per sensor, so the ridge is fitted on one row per sensor, with its penalty chosen by leave-one-out over the sensors. A held-out sensor is predicted from its network terms alone, the same way the candidates predict it.

Each candidate is trained in two variants, which differ only in features (`src/features/schema.py` marks each one):
- `rich`: every feature, including the Melbourne-only layers (CLUE jobs and floor space, City business data, canopy, the footpath-steepness survey).
- `portable`: only features any city has (OSM network and amenities, DEM slope, shade, weather, calendars, events). This is the variant for NYC and Boston.

The baselines use only network features, which both variants share, so they run once.

## Folds

Rows and features come from the training frame (`src/features/training_frame.py`, `docs/features/training_frame.md`): located outdoor sensors, outage hours dropped. Folds come from its `splits.parquet`:
- sensors within 200 m of each other share a spatial group, and a group is never split between training and testing;
- the three holdout precincts (UniMelb edge, Carlton, North Melbourne) and their neighbours are in no fold, so no model trains on them before the final evaluation (CLAUDE.md core rule 2).

Each fold's training rows are a 600,000-row sample of the other folds, drawn once per fold, so every candidate and baseline trains on exactly the same rows. Every model then predicts its fold's sensors, which gives out-of-fold predictions for every fold sensor.

## Metrics

All scoring is in `src/models/flow/evaluate.py`, named as in `experiment_tracking.md`. Every model is scored on the hours after each sensor's calibration window (`scored_mask`), so calibrated and uncalibrated scores cover the same hours. `*_fold<k>` is one location fold and `*_cv` is the mean over the folds (`cv_metrics`, which refuses holdout rows).

| Metric | Meaning |
|---|---|
| `log_mae_cv` | MAE on log1p counts: the success criterion and the selection metric |
| `wape_cv`, `wape_fold0..4` | Sum of absolute errors over sum of counts, overall and per fold |
| `mae_cv`, `rmse_cv` | On raw counts |
| `bias_cv` | Predicted total over counted total, minus one |
| `sensor_rank_corr_cv` | Spearman correlation of sensors' mean predicted and mean counted volume |
| `log_mae_cv_cal`, `wape_cv_cal` | After scaling each sensor to its first 28 days of counts (`calibrated_cv_metrics`) |
| `coverage80_cv`, `coverage80_fold<k>` | Share of counts inside the widened 10 to 90% interval (`cv_coverage`; selected model only) |
| `coverage80_raw_cv` | The same for the quantile models' interval before widening |
| `interval_margin_low`, `interval_margin_high` | The widening, on log1p counts (also in `selected.json`) |
| `interval_width_ratio` | Median of (1 + upper) / (1 + lower) for the widened interval |
| `log_mae_holdout`, `wape_holdout` | On the holdout precincts, once (selected model only) |
| `log_mae_<part>`, `wape_<part>` | Event days, other days and each era of `configs/bias.yaml`, scored apart (`standard_parts`, `score_parts`; selected model only) |
| `worst_slice_ratio` | Worst bias-slice WAPE over overall WAPE (selected model only) |

## Hyperparameters

Defaults live in `configs/models.yaml`. With `--tune`, an Optuna TPE search (30 trials) runs per family and variant before training. Trials are scored by log MAE on two inner folds over the fold sensors, split by spatial group, on 200,000-row samples. Each trial is a nested MLflow run under `tune_<family>_<variant>`, and the parent logs the best values and Optuna's estimate of each parameter's importance (hyperparameter sensitivity).

| Family | Searched |
|---|---|
| hgb | `learning_rate` 0.03 to 0.15 (log), `max_iter` {200, 400}, `max_leaf_nodes` 31 to 127, `min_samples_leaf` 50 to 300, `l2_regularization` 0.5 to 5 (log) |
| lgbm | `learning_rate` 0.03 to 0.15 (log), `n_estimators` {200, 400}, `num_leaves` 31 to 127, `min_child_samples` 50 to 300, `reg_lambda` 0.5 to 5 (log), `colsample_bytree` 0.5 to 1 |
| xgb | `learning_rate` 0.03 to 0.15 (log), `n_estimators` {200, 400}, `max_depth` 4 to 10, `min_child_weight` 1 to 100 (log), `reg_lambda` 0.5 to 5 (log), `colsample_bytree` 0.5 to 1 |

The inner folds draw on the same sensors the `*_cv` metrics score, so tuned `*_cv` numbers are slightly optimistic. Holdout precincts are never seen. Untuned runs (the monthly default) don't have this issue.

## Selection, after the bias check

`src.models.flow.select`, per variant:
1. Rank the candidates by `log_mae_cv`.
2. Shortlist those within 3% of the best.
3. For each shortlisted candidate, measure WAPE per slice of its out-of-fold predictions. The slices are the ones in `configs/bias.yaml` (time of day, rain, temperature, slope, precinct, era, volume and amenity tiers, AFL days and their crosstabs), and the code is `src/bias`'s: Fairlearn `MetricFrame` with sensor-bootstrap intervals. A candidate's worst slice is its largest group-to-overall WAPE ratio, among groups with at least five sensors.
4. The candidate with the smallest worst slice wins, so a slightly more accurate model can't win while failing one group.

Then, for the winner:
- 10% and 90% quantile models of the same family are trained on the same folds, and their interval is widened from its out-of-fold misses (conformalized quantile regression, below), which gives `coverage80_cv`;
- the model and its two interval models are fitted on up to 2 million rows from every fold sensor;
- the holdout precincts are scored for the first time: uncalibrated, and calibrated on one or two of each precinct's sensors (one scale from their first 28 days, applied to the others). The UniMelb edge is split at the Metro Tunnel opening;
- the calibration curve: one site-wide scale from k = 1, 3, 5, 10, 20 random sensors, applied to the rest, 20 draws each. In Melbourne this mostly shows that one scale can't fix errors that differ by sensor: the model trained there, so there is no city-wide offset to remove. It's meant for a new city such as NYC, where there is one;
- feature importance: mean |SHAP| (TreeExplainer) on a 2,000-row sample, with permutation importance as the fallback.

The selected run (`stoa.role = selected`) holds the three models, `features.json`, and a `selection/` folder: `ranking.csv`, `slices.csv`, `bias_check.csv`, `holdout_precincts.csv`, `parts.csv`, `calibration_curve.csv` and `feature_importance.csv`, with charts of the model comparison, the calibration curve and the feature importance.

### Prediction intervals

The quantile models learn the spread of counts at the sensors they trained on. At a new location most of the error is in the sensor's overall level, which can be off by a factor of three either way, so their raw interval covers far less than 80% of out-of-fold counts. `select.interval_margins` measures, on log1p counts, how far out-of-fold counts fall below the 10% bound and above the 90% bound. It picks each side's margin so that 10% of counts stay outside that side. Prediction widens the interval by those margins (`select.widen`) using the values in `selected.json`. `coverage80_cv` widens each fold with margins from the other folds only, so the margins never saw the sensors they are scored on.

The widened interval describes a location with no local counts. After calibrating on a site's own counts, most of the level error is gone and the interval is conservative.

### Training reports

Every model trained gets a Markdown report (`src/models/flow/report.py`), written to `data/models/<variant>/reports/<model>.md` (the baselines' to `data/models/baselines/reports/`) and logged to its MLflow run as `model_report.md`. Each run of `train` and `select` rewrites them. A report lists:
- the data: the period of the training rows, how many rows and sensors, and which sensors the model never saw (each fold's, and the holdout precincts);
- the loss function (per row: Poisson deviance for the mean models, pinball loss for the interval models, squared error on log counts for the baselines) and the cost function (its mean over a set of rows), with the cost before the first tree and after the last, on training rows and on rows the model never saw. Boosted models also add an L2 penalty on leaf values during training, which the reported costs leave out so they compare across families;
- the learned parameters: trees, leaves, splits and features used (for the baselines, the hour-of-week profile and the ridge coefficients);
- the hyperparameters, and whether they were tuned and over what search space.

The reports aren't tracked in git, because writing tracked files during a run would mark the run's later models as trained on uncommitted code. Copy them into `docs/` for a submission.

After selection, `src.models.flow.mitigate` retrains the selected model with sensor-equal and volume-tier weights and calibrates it, all scored on the same slices. Results, the trade-offs and the gate's bias limit are in [model_bias.md](../bias/model_bias.md).

## Gate, registry and rollback

`src.models.flow.registry --check` compares the latest selected run of each variant with the latest `baseline_space_syntax` run on the same data version (`stoa.dvc.processed`) and fold plan (`stoa.plan`). A variant passes when:
- its `log_mae_cv` is at least 15% below Space Syntax's;
- its `coverage80_cv` is between 0.75 and 0.85;
- its worst slice is under `gates.max_worst_slice_ratio`, 2.25 times the overall WAPE ([model_bias.md](../bias/model_bias.md)).

If `rich` fails, the command exits 1 and nothing is registered. `portable` is registered only if it passes. The models of 12 November 2026 miss the 15% (11.7% rich, 10.3% portable); what we tried to close the gap, and the recommendation to the team, are in [accuracy_target.md](accuracy_target.md).

Because the gate reads the latest selected run, an experiment logged to `stoa-flow` becomes the run the gate checks. Set `MLFLOW_EXPERIMENT_NAME` for anything that isn't a candidate for release.

Each variant is its own registered model, `stoa-flow-rich` and `stoa-flow-portable`, and versions are tracked with aliases rather than MLflow's deprecated stages:
- `--promote` points `champion` (the served version) at the new version, and `previous` at the one it replaced;
- without `--promote`, the new version gets `challenger`;
- `--rollback --variant rich` points `champion` back at `previous`.

Each command writes what it did to `data/models/registry.json`.

## Outputs

| Where | What |
|---|---|
| MLflow, experiment `stoa-flow` | Every candidate, baseline, tuning and selected run, with code and data versions |
| `data/models/<variant>/oof.parquet` | Out-of-fold predictions of every model, one column each |
| `data/models/<variant>/candidates.json` | Each model's run id, parameters, metrics and the fold plan |
| `data/models/<variant>/selected.json`, `selected/` | The winner, its model URIs, metrics and selection tables |
| `data/models/registry.json` | The last gate check, registration or rollback |

`data/models/` is git-ignored and not DVC-tracked: the models and their history live in MLflow.

## Commands

```bash
docker compose up -d mlflow                     # the tracking server (localhost:5001)
.venv/bin/python -m src.models.flow.train       # candidates and baselines, both variants
.venv/bin/python -m src.models.flow.select      # bias check, intervals, final fit, holdouts
.venv/bin/python -m src.models.flow.registry --check     # gates only
.venv/bin/python -m src.models.flow.registry --promote   # gate, register, champion
```

Quick runs: `train --folds 2 --max-trees 40 --train-rows 200000` and `select --final-train-rows 300000`. `train --tune` runs the search first, and `--families hgb lgbm` limits the candidates.

`train --train-start 2022-01-01` trains only on hours from that date (`flow.train_start`). Each fold's rows are all still predicted and scored, and `select` and `mitigate` read the start from `candidates.json`. `train --params-from data/models` reuses each family's parameters from an earlier run instead of the defaults. The comparison of start years is in [training_periods.md](training_periods.md).

In Airflow, the `stoa_train` DAG runs the same four steps monthly or on demand (`tune` param), after a good `stoa_pipeline` run. The daily pipeline never trains.

## Not included yet

Ladder rung 3 in `docs/PROJECT_CONTEXT.md`, hand-weighted costs, is left out for now, so the ladder runs from betweenness to Space Syntax to the learned candidates. There is no random forest candidate either. The NYC cross-city test is a separate step: it needs the portable model's predictions joined to NYC's count periods.
