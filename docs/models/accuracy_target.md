# The 15% accuracy target

The team's success criterion (October 2026, `docs/PROJECT_CONTEXT.md`) was that the flow model's log MAE on held-out sensors be at least 15% lower than the Space Syntax baseline's.

**Decision (16 November 2026):** the team lowered the gate to **10% below Space Syntax** (`flow.gates.log_mae_improvement: 0.10` in `configs/models.yaml`), the first of the two options recommended below. The write-up reports the 15% target as missed, with the results on this page.

**Where it stands (15 November 2026).** On the buffered location folds, the rich model is 14.3% to 15.5% below Space Syntax across three seeds (mean 14.8%), and the portable model 13.1% to 13.8% (mean 13.5%). Resampling the spatial groups gives each run a 95% range of roughly 5% to 25%. The 15% target sits inside that noise: one seed passes it and two don't.

**What we recommended.** The gate shouldn't depend on the seed. The two options were:
- lower the gate to 10%, which every buffered run clears (the lowest is 11.5%, the portable run retuned on two inner folds), and report the 15% target as missed by a small margin, at 14.8% on the mean of three seeds; or
- keep 15% and gate on the mean of several seeds, which costs about 20 minutes of training per extra seed.

The team chose the first.

## The neighbour buffer

Until 15 November the location folds grouped sensors within 200 m of each other (complete linkage), but two sensors a little over 200 m apart could still land in different groups, one in training and one in testing. Arsh found these neighbours, which made the held-out score closer to "an uncounted spot in a city we've measured" than to "a new area". Each fold now also drops from its training rows every sensor within 200 m of one of its test sensors (`buffer_m` in `configs/models.yaml`; `close_sensors` and `fold_plan` in `src/models/flow/data.py`), and the tuning folds do the same.

The buffer barely moved the model and made the baseline worse. Both runs below used the same parameters (the tuning search of 12 November) and seed 7:

| | Space Syntax | Rich (lgbm) | Below Space Syntax | Portable (lgbm) | Below Space Syntax |
|---|---|---|---|---|---|
| Unbuffered (12 November) | 0.979 | 0.864 | 11.7% | 0.878 | 10.3% |
| Buffered (15 November) | 1.023 | 0.865 | 15.5% | 0.883 | 13.7% |

Space Syntax predicts from a few network metrics and an hour-of-week profile fitted on the training sensors, so it likely leaned on a next-door training sensor more than the model did. The buffered numbers answer the question the project asks (how well does the model do in an area with no counts?), so they're the ones to report.

## The buffered runs

All on data version `961075e6`, in the MLflow experiment `stoa-flow-buffer`, so the main `stoa-flow` runs and the gate are untouched. log MAE on the five location folds:

| Run | Space Syntax | Rich: selected | log MAE | Below Space Syntax | Portable: selected | log MAE | Below Space Syntax |
|---|---|---|---|---|---|---|---|
| Seed 7, 12 November parameters | 1.023 | lgbm | 0.865 | 15.5% | lgbm | 0.883 | 13.7% |
| Seed 8, same | 1.023 | xgb | 0.874 | 14.6% | xgb | 0.889 | 13.1% |
| Seed 9, same | 1.023 | lgbm | 0.877 | 14.3% | xgb | 0.882 | 13.8% |
| Seed 7, retuned on two buffered inner folds | 1.023 | lgbm | 0.886 | 13.4% | lgbm | 0.906 | 11.5% |
| Seed 7, retuned on four buffered inner folds (rich, lgbm and xgb only) | 1.023 | lgbm | 0.866 | 15.4% | | | |

The seed changes each fold's 600,000-row training sample and the models' own randomness, not the folds. Every selected model beats betweenness (1.090) by 17% to 21%.

**How sure is the gain?** `log_mae_cv` averages the five folds. Pooling every scored hour instead (each sensor's hours after its first 28 days), and resampling the 49 spatial groups 2,000 times:

| Run | Rich: pooled gain | 95% range | Sensors where the model beats Space Syntax | Portable: pooled gain | 95% range |
|---|---|---|---|---|---|
| Seed 7 | 17.0% | 9.1% to 24.2% | 82 of 104 | 15.4% | 5.7% to 23.5% |
| Seed 8 | 16.4% | 5.2% to 25.2% | 74 of 104 | 14.7% | 4.7% to 23.2% |
| Seed 9 | 15.3% | 6.6% to 23.1% | 73 of 104 | 15.5% | 5.7% to 24.3% |
| Unbuffered, 12 November | 13.3% | 4.9% to 20.9% | 72 of 104 | 11.3% | 3.1% to 19.3% |

The pooled gain differs from the fold average because folds with more hours count for more. With only 49 groups, which ones a resample draws moves the gain by several points, so the ranges are wide.

**The retuned run is worse.** Tuning on buffered inner folds chose more cautious settings than the search of 12 November (LightGBM learning rate 0.034 against 0.13; XGBoost trees 4 deep against 7). Those settings scored worse on the outer folds, on the holdout precincts (rich 1.148 against 0.881) and after calibration (0.664 against 0.603). The tuning uses two inner folds, so each trial trains on about half the fold sensors, minus their neighbours, and that seems to reward models that underfit. To test that, we reran the search with four inner folds (rich variant, LightGBM and XGBoost only, in the MLflow experiment `stoa-flow-buffer-inner4`). It picked settings close to the 12 November ones (LightGBM learning rate 0.075, 96 leaves, at least 136 rows per leaf; XGBoost trees 7 deep) and scored as well as them: LightGBM 0.866 and XGBoost 0.866, both 15.4% below Space Syntax, with calibrated error 0.606. On the holdout precincts it scored 0.958, between the two-fold search (1.148) and the 12 November parameters (0.881). `tune.inner_folds` is now 4. Each trial fits four models instead of two, so a search takes longer: 38 minutes for these two families on one variant. The monthly `stoa_train` run doesn't search anyway: without `tune` it reuses the last run's parameters (`--params-from`).

**Larger leaves, buffered.** The same leaf-size test as before (below), on the seed-7 buffered folds, rich variant:

| LightGBM leaf minimum (rows) | log MAE | Level from this model, shape from the selected one |
|---|---|---|
| 130 (selected) | 0.865 | |
| 1,000 | 0.859 | 0.860 |
| 3,000 | 0.857 | 0.853 |
| 5,000 | 0.866 | 0.859 |
| 10,000 | 0.859 | 0.842 |

Plain larger leaves gain at most 1%. The level-plus-shape split at 10,000 rows reaches 0.842 (17.7% below Space Syntax), but it still isn't deployable as it stands (a segment's level would need a fixed set of reference hours to average over), and the leaf size was picked on these same folds. The portable variant behaves the same way: plain leaves reach 0.870 at best and the split 0.858, against 0.883.

## What we tried before the buffer

Everything in this section ran on the unbuffered folds of 12 November, rich variant, so the absolute numbers are out of date. The conclusions about what doesn't help still hold.

- **Training on log counts** instead of Poisson. Every loss was worse: squared error 0.918, absolute error 0.905, Huber 0.890, against Poisson's 0.864.
- **Blending** the three model families, or mixing some of the Space Syntax prediction into the model. No gain: the models miss the same sensors by about the same factor, so averaging doesn't cancel the errors.
- **Larger leaves.** Leaf minimums of 1,000, 3,000, 5,000, 10,000 and 30,000 rows, hand-set, gave 0.874, 0.858, 0.844, 0.846 and 0.908. Calibrated error got worse as leaves grew, because big leaves blur the hourly pattern that calibration relies on. Level from a 10,000-row model with shape from the selected one reached 0.834.
- **A fair test of larger leaves** (branch `romit/leaf-size`). Tuning searched leaf minimums up to 20,000 rows and picked about 3,100. Rich got worse (0.882). Portable improved on the folds (0.860) but got much worse on the holdout precincts (1.221 against 1.082), most of all at the UniMelb edge. We don't recommend merging it. Its runs are in `stoa-flow` (commit 59b1242).
- **Later training starts** ([training_periods.md](training_periods.md)). Every start after 2009 is worse on all hours.

## Why the gap is hard to close

Most of the error is in each sensor's overall level, not in its daily and weekly pattern. Calibrating on a sensor's first four weeks of counts cuts the rich model's log MAE from 0.865 to 0.603, and its sensor rank correlation rises from 0.53 to 0.95. A place's absolute volume probably depends on things the features don't capture, such as a busy building entrance next to the sensor or exactly where it sits on the footpath. The project's generalization plan expects this: the general model learns relative patterns and local counts set the volume (CLAUDE.md, "Generalization").

The CLAUDE.md checkpoint is to stop investing in the model if it doesn't beat Space Syntax. It does, so the project carries on as planned.

## What changed with the decision

- `flow.gates.log_mae_improvement` is 0.10, and the criterion is updated in `docs/PROJECT_CONTEXT.md`, `docs/models/flow_model.md` and `docs/models/modelling_plan.md`.
- `stoa-flow` still holds the unbuffered models of 12 November. The next `stoa_train` run on main trains on buffered folds with their parameters (it reuses the last run's, `--params-from`), gates against 10%, and registers and promotes the models that pass.
- The 15% target is reported as missed, with these results, not dropped from the write-up.
