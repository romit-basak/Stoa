# Bias in the flow model: slices, mitigation and the gate

This covers the flow model itself, from the buffered run of 15 November 2026 (seed 7, the 12 November parameters, data version `961075e6`, LightGBM selected in both variants; MLflow experiment `stoa-flow-buffer`). The data-stage bias work, with a stand-in model, is in [bias_slicing.md](bias_slicing.md); this page uses the same slices and code on the real model. The unbuffered models of 12 November gave the same picture with slightly different numbers.

## How bias is checked

1. **Slices.** Every out-of-fold prediction is grouped by the slices in `configs/bias.yaml`: time of day, rain, temperature, slope, precinct, era, volume tier, amenity tier, missing hours, public holidays, AFL days and their crosstabs. Each prediction comes from a model that never saw that sensor or any sensor within 200 m of it. Each sensor's first 28 days are left out, so the check scores the same hours as the calibrated metrics.
2. **Metric.** WAPE per group with Fairlearn's `MetricFrame`, with a 95% interval from resampling sensors (300 resamples). A group is **worse** when its whole interval is above the overall WAPE. Its **ratio** is its WAPE over the overall WAPE. The **worst slice** is the largest ratio among groups with at least 5 sensors.
3. **Selection after the bias check** (`src/models/flow/select.py`). Candidates within 3% of the best log MAE are compared on their worst slice, and the fairest wins. For the rich variant that chose LightGBM (log MAE 0.865, worst slice 1.70) over XGBoost (0.864, 1.77).
4. **Mitigation** (`src/models/flow/mitigate.py`). The selected model is retrained on the same folds with each weighting of the training rows, or calibrated, and scored on the same slices and the same hours.
5. **Gate** (`src/models/flow/registry.py`). A selected model whose worst slice is over 2.25 times the overall WAPE isn't registered. In `stoa_train`, that fails the `check_gates` task, which emails the team.

## What the slices show

For the selected rich model, six groups are clearly worse than overall (WAPE 0.632):

| Group | WAPE | Ratio | Sensors |
|---|---|---|---|
| Quiet sensors | 1.08 | 1.70 | 29 |
| Southbank | 0.83 | 1.29 | 6 |
| Low amenity | 0.74 | 1.15 | 33 |
| Public holidays | 0.70 | 1.10 | 103 |
| Night | 0.69 | 1.07 | 104 |
| Carlton | 0.78 | 1.22 | 1 (too thin for the worst-slice ratio) |

The portable model has the same pattern, with nine groups worse. Its worst are quiet sensors (1.67 times, 29 sensors) and Carlton (1.55 times, on its one sensor), then Southbank, low amenity, the City's "Remainder" precinct, holidays, the COVID era and night.

Quiet sensors aren't missed by a bigger factor than busy ones. On the 12 November models, the median miss in a sensor's overall volume was 1.82 times either way at quiet sensors and 1.89 times at busy ones. But WAPE divides by counts, and hour-to-hour noise is a larger share of a small count. Most of their error is in the overall level, not the daily pattern.

## Mitigations tried

Rich variant (portable in brackets where it differs):

| Mitigation | log MAE | WAPE | Worst slice | Groups worse | Quiet | Night | COVID |
|---|---|---|---|---|---|---|---|
| None (selected model) | 0.865 | 0.632 | 1.70, quiet (1.67) | 6 (9) | 1.08 | 0.69 | 0.68 |
| Sensor-equal weights | 0.899 | 0.660 | 1.56, quiet (1.67) | 6 (9) | 1.02 | 0.69 | 0.70 |
| Volume tiers weighted equally | 0.895 | 0.652 | 1.65, quiet (1.58) | 5 (9) | 1.09 | 0.70 | 0.69 |
| Calibrated to 28 days of local counts | 0.603 | 0.420 | 1.69, Melbourne (Remainder) (1.63) | 15 (15) | 0.57 | 0.65 | 0.67 |

- **Reweighting buys a little fairness for a lot of accuracy.** Weighting sensors equally lowers the rich model's worst slice from 1.70 to 1.56, but its log MAE rises 4% (0.865 to 0.899) and its overall WAPE rises too, while the quiet sensors only improve from 1.08 to 1.02. For the portable model it changes nothing. Weighting the volume tiers equally helps the portable model's worst slice a little (1.67 to 1.58) at a similar cost. Weights shift which rows the trees fit, but the quiet sensors' problem is their level, which no row weighting fixes.
- **Calibration does help.** Scaling each sensor to its first 28 days of counts halves the quiet sensors' error (1.08 to 0.57) and cuts the overall WAPE from 0.632 to 0.420.
- **What's left after calibration is a different problem.** More groups now count as "worse", but only because the overall error they're compared with fell so far: 59 of the 60 groups improve in absolute terms. The exception is Parkville, on one sensor. The worst slices are now the City's "Remainder" precinct (1.69 times), the COVID era and night (0.65, 1.58 times), and Docklands. One scale per sensor can't fix a pattern that's wrong at some hours; those need changes to the model's features, not weights or scales.

## Decision and trade-offs

- **Keep the selected model unweighted.** Both weightings cost accuracy, and the fairness they buy is small and not consistent across the two variants.
- **Calibration is the mitigation.** That's the product's design anyway: a site with even a few weeks of local counts gets calibrated predictions (CLAUDE.md, "Generalization"). The trade-off is that it needs local counts. A site without any keeps the uncalibrated model's errors.
- **For uncalibrated sites, say so in the app.** Predictions on quiet streets can be off by more than their own volume. Relative comparisons ("busier than", rankings, changes under a scenario) hold up better than absolute counts there. The intervals already widen for this.
- **The gate limit is 2.25.** It was set just above the 12 November XGBoost candidate's 2.24. Every buffered run so far has a worst slice between 1.67 and 1.90, so a retrain that makes the worst slice noticeably worse is blocked and emailed, without failing on today's models. It's a team setting in `configs/models.yaml` (`gates.max_worst_slice_ratio`).
- **Next, for night and COVID:** features that describe night-time activity (venues open late, lighting), and a check on whether the COVID era should be down-weighted or dropped. Training from later start years didn't help ([training_periods.md](../models/training_periods.md)).

## Running it

```
.venv/bin/python -m src.models.flow.select      # selection, after the bias check
.venv/bin/python -m src.models.flow.mitigate    # the mitigations above, about 17 minutes
.venv/bin/python -m src.models.flow.registry --check
```

`mitigate` writes `data/models/<variant>/mitigation/summary.csv`, `slices.csv` (every group under every mitigation) and `mitigation.png`. Each mitigation is an MLflow run, `mitigation_<name>_<variant>`, with `stoa.role = mitigation`. The mitigations are listed under `flow.mitigation` in `configs/models.yaml`. `stoa_train` runs the step between selection and the gate.
