# selected: lgbm_portable

The selected flow model for the portable variant: `lgbm` with a Poisson objective, and `lgbm` quantile models at 0.1, 0.9 for the interval (widened by `interval_margins` in `selected.json`).

Trained 2026-11-16T05:45:52+00:00.

## Data

- Period: 2009-05-01 to 2026-11-15 (local dates of the training rows)
- Training rows: 2,000,000 (sampled)
- Sensors: 104 trained on, of 118 in the frame
- Never seen, holdout precincts and their neighbours: 14 sensors (31, 37, 42, 43, 44, 46, 50, 54, 70, 76, 85, 86, 87, 180)
- Trained on every fold sensor; the holdout precincts are scored here for the first time.

## Loss and cost

- Loss function (per row): Poisson deviance for the mean model; pinball loss for the interval models
- Cost function: the mean loss over the rows (sampled, 20,000 per set)

**Mean model (Poisson deviance)**

| Rows | Start | End | Change |
|---|---|---|---|
| training rows | 780.2 | 58.56 | -92.5% |
| holdout precincts | 536.5 | 144.3 | -73.1% |

By trees (0 is before the first tree):

| trees | training rows | holdout precincts |
|---|---|---|
| 0 | 780.2 | 536.5 |
| 4 | 545.8 | 504.3 |
| 40 | 133.4 | 293.4 |
| 100 | 80.24 | 186.8 |
| 200 | 65.99 | 151.7 |
| 400 | 58.56 | 144.3 |

**10% interval model (pinball loss)**

| Rows | Start | End | Change |
|---|---|---|---|
| training rows | 50.3 | 24.06 | -52.2% |
| holdout precincts | 14.71 | 161.6 | +998.5% |

By trees (0 is before the first tree):

| trees | training rows | holdout precincts |
|---|---|---|
| 0 | 50.3 | 14.71 |
| 4 | 48.18 | 13.39 |
| 40 | 35.35 | 39.65 |
| 100 | 28.58 | 96.64 |
| 200 | 25.89 | 151.1 |
| 400 | 24.06 | 161.6 |

**90% interval model (pinball loss)**

| Rows | Start | End | Change |
|---|---|---|---|
| training rows | 182.2 | 30.58 | -83.2% |
| holdout precincts | 128.9 | 69.51 | -46.1% |

By trees (0 is before the first tree):

| trees | training rows | holdout precincts |
|---|---|---|
| 0 | 182.2 | 128.9 |
| 4 | 123.5 | 111.7 |
| 40 | 41.04 | 76.63 |
| 100 | 35.51 | 72.36 |
| 200 | 32.72 | 70.92 |
| 400 | 30.58 | 69.51 |

## Learned parameters

- model: trees: 400
- model: features in: 152
- model: leaves: 44000
- model: splits: 43600
- model: learned values: 87600
- model: features used: 137
- model q10: trees: 400
- model q10: features in: 152
- model q10: leaves: 44000
- model q10: splits: 43600
- model q10: learned values: 87600
- model q10: features used: 135
- model q90: trees: 400
- model q90: features in: 152
- model q90: leaves: 44000
- model q90: splits: 43600
- model q90: learned values: 87600
- model q90: features used: 137

## Hyperparameters

Defaults from `configs/models.yaml` (not tuned in this run).

- `colsample_bytree`: 0.5414
- `learning_rate`: 0.1122
- `min_child_samples`: 184
- `n_estimators`: 400
- `num_leaves`: 110
- `reg_lambda`: 2.941
- `subsample`: 0.8
- `subsample_freq`: 1

## Scores

- `log_mae_cv`: 0.8877
- `wape_cv`: 0.6448
- `wape_cv_cal`: 0.419
- `coverage80_cv`: 0.78
- `log_mae_holdout`: 1.017
- `wape_holdout`: 1.005
- `coverage80_holdout`: 0.6002
