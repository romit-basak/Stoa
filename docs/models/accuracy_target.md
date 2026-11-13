# The 15% accuracy target

The team's success criterion (October 2026, `docs/PROJECT_CONTEXT.md`) is that the flow model's log MAE on held-out sensors be at least 15% lower than the Space Syntax baseline's. The registry gate enforces it (`flow.gates.log_mae_improvement: 0.15` in `configs/models.yaml`). With Space Syntax at 0.979, the model needs 0.832 or lower.

The models of 12 November 2026 fall short:

| Variant | Selected model | log MAE | Below Space Syntax | Below betweenness (1.073) |
|---|---|---|---|---|
| Rich | lgbm | 0.864 | 11.7% | 19.5% |
| Portable | lgbm | 0.878 | 10.3% | 18.2% |

Both pass the other gate checks (interval coverage 0.776 and 0.783; worst slice 1.90 and 1.94).

**Recommendation:** the team should accept "beats both baselines" for Tier 1, or lower the gate to 10% against Space Syntax, and record that as a team decision. We tried the changes most likely to close the gap. None reached 15% in a fair test, and the fair test that came closest made the held-out precincts worse. Tuning further against the same folds would mean fitting the gate rather than improving the model.

## What we tried

All numbers are out-of-fold log MAE for the rich variant on the gate's own five location folds, unless stated otherwise.

**Training on log counts** instead of Poisson. Every loss was worse: squared error 0.918, absolute error 0.905, Huber 0.890, against Poisson's 0.864.

**Blending** the three model families, or mixing some of the Space Syntax prediction into the model. No gain. The models miss the same sensors by about the same factor, so averaging them doesn't cancel the errors.

**Larger leaves.** Each fold trains on about 6,000 rows per sensor, but tuning only searched leaf minimums of 50 to 300 rows. Leaves that small can fit each training sensor's own level, which does not carry over to a new sensor. Forcing larger leaves, hand-set, gave:

| LightGBM leaf minimum (rows) | log MAE | WAPE | Sensor rank correlation | log MAE after calibration |
|---|---|---|---|---|
| 1,000 | 0.874 | 0.652 | 0.555 | 0.597 |
| 3,000 | 0.858 | 0.633 | 0.514 | 0.598 |
| 5,000 | 0.844 | 0.606 | 0.617 | 0.605 |
| 10,000 | 0.846 | 0.617 | 0.591 | 0.624 |
| 30,000 | 0.908 | 0.631 | 0.556 | 0.698 |

The best of these is 13.8% below Space Syntax. Calibrated error gets worse as leaves grow, because big leaves blur the hourly pattern that calibration relies on.

**Level from a big-leaf model, pattern from the current one.** We took each sensor's overall level from the big-leaf model and its hour-by-hour shape from the current model. At 10,000 rows this reached 0.834 (14.8%) and kept calibrated error at 0.597. It is not deployable as it stands: a segment's level would need a fixed set of reference hours to average over, which the prediction API doesn't have. These leaf sizes were also chosen on the gate's own folds, so all the numbers in this section are optimistic.

**A fair test of larger leaves.** On branch `romit/leaf-size`, leaf minimums are counted per 600,000-row fold sample and rescaled for the smaller tuning folds and the larger final fit. Tuning searched 50 to 20,000 rows. We then ran the full train, select, mitigate and gate. Tuning picked LightGBM leaf minimums of about 3,100 to 3,200 rows for both variants:

| | Rich, current | Rich, leaf search | Portable, current | Portable, leaf search |
|---|---|---|---|---|
| Selected family | lgbm | xgb | lgbm | lgbm |
| log MAE | 0.864 | 0.882 | 0.878 | 0.860 |
| Below Space Syntax | 11.7% | 9.9% | 10.3% | 12.1% |
| Holdout precincts, log MAE | 0.874 | 0.930 | 1.082 | 1.221 |
| Holdout precincts, WAPE | 0.830 | 0.957 | 1.054 | 1.490 |
| UniMelb edge after the Metro Tunnel, log MAE | 1.019 | 1.130 | 1.165 | 1.746 |

The tuned rich LightGBM scored 0.859 (12.2%), but its worst slice (quiet sensors) was 2.37 times the overall WAPE, over the 2.25 limit. Selection therefore took XGBoost, which is worse. Portable improved on the folds but got much worse on the held-out precincts, the within-Melbourne transfer test. Big leaves pull an unusual place toward a typical level, and the UniMelb edge, the most campus-like precinct, lost the most. We don't recommend merging `romit/leaf-size`. Its runs are in the `stoa-flow` experiment (commit 59b1242) for reference.

**Later training starts** (`docs/models/training_periods.md`). Every start after 2009 is worse on all hours, so this doesn't help either.

## Why the gap is hard to close

Most of the error is in each sensor's overall level, not in its daily and weekly pattern. Calibrating on a sensor's first four weeks of counts cuts the rich model's log MAE from 0.864 to 0.595. A place's absolute volume probably depends on things the features don't capture, such as a busy building entrance next to the sensor or exactly where it sits on the footpath. The project's generalization plan already expects this. The general model learns relative patterns and per-site calibration sets the volume (CLAUDE.md, "Generalization"), and the calibration curve is meant to be the headline result.

The CLAUDE.md checkpoint is to stop investing in the model if it doesn't beat Space Syntax. It does beat Space Syntax, so the project carries on as planned.

## If the team agrees

- Set `flow.gates.log_mae_improvement` to the agreed value (for example 0.10) in `configs/models.yaml`, and update the criterion in `docs/PROJECT_CONTEXT.md`, `docs/models/flow_model.md` and `docs/models/modelling_plan.md`.
- Run `registry --check`, then `--promote` once both variants pass.
- Report the 15% target as missed, with these results, rather than dropping it from the write-up.
