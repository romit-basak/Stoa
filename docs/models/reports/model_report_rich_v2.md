# selected: lgbm_rich

The selected flow model for the rich variant: `lgbm` with a Poisson objective, and `lgbm` quantile models at 0.1, 0.9 for the interval (widened by `interval_margins` in `selected.json`).

Trained 2026-11-16T05:28:58+00:00.

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
| training rows | 780.2 | 55.42 | -92.9% |
| holdout precincts | 536.5 | 122.7 | -77.1% |

By trees (0 is before the first tree):

| trees | training rows | holdout precincts |
|---|---|---|
| 0 | 780.2 | 536.5 |
| 4 | 500.9 | 498.9 |
| 40 | 121 | 236.3 |
| 100 | 74.94 | 153.6 |
| 200 | 62.72 | 131.1 |
| 400 | 55.42 | 122.7 |

**10% interval model (pinball loss)**

| Rows | Start | End | Change |
|---|---|---|---|
| training rows | 50.3 | 23.79 | -52.7% |
| holdout precincts | 14.71 | 123 | +736.2% |

By trees (0 is before the first tree):

| trees | training rows | holdout precincts |
|---|---|---|
| 0 | 50.3 | 14.71 |
| 4 | 47.99 | 13.19 |
| 40 | 35.32 | 25.54 |
| 100 | 28.66 | 74.81 |
| 200 | 25.58 | 109.2 |
| 400 | 23.79 | 123 |

**90% interval model (pinball loss)**

| Rows | Start | End | Change |
|---|---|---|---|
| training rows | 182.2 | 30.24 | -83.4% |
| holdout precincts | 128.9 | 61.82 | -52.1% |

By trees (0 is before the first tree):

| trees | training rows | holdout precincts |
|---|---|---|
| 0 | 182.2 | 128.9 |
| 4 | 114.4 | 110.9 |
| 40 | 40.04 | 68.48 |
| 100 | 34.91 | 64.65 |
| 200 | 32.15 | 63.09 |
| 400 | 30.24 | 61.82 |

## Learned parameters

- model: trees: 400
- model: features in: 163
- model: leaves: 37600
- model: splits: 37200
- model: learned values: 74800
- model: features used: 147
- model q10: trees: 400
- model q10: features in: 163
- model q10: leaves: 37600
- model q10: splits: 37200
- model q10: learned values: 74800
- model q10: features used: 147
- model q90: trees: 400
- model q90: features in: 163
- model q90: leaves: 37600
- model q90: splits: 37200
- model q90: learned values: 74800
- model q90: features used: 148

## Hyperparameters

Defaults from `configs/models.yaml` (not tuned in this run).

- `colsample_bytree`: 0.5141
- `learning_rate`: 0.1324
- `min_child_samples`: 130
- `n_estimators`: 400
- `num_leaves`: 94
- `reg_lambda`: 4.526
- `subsample`: 0.8
- `subsample_freq`: 1

## Scores

- `log_mae_cv`: 0.8775
- `wape_cv`: 0.6386
- `wape_cv_cal`: 0.4241
- `coverage80_cv`: 0.7722
- `log_mae_holdout`: 0.878
- `wape_holdout`: 0.8389
- `coverage80_holdout`: 0.6447
