# Bias in the flow model: slices, mitigation and the gate

This covers the flow model itself, from the training on 12 November 2026 (data version `9b0ab86`, LightGBM selected in both variants). The data-stage bias work, with a stand-in model, is in [bias_slicing.md](bias_slicing.md); this page uses the same slices and code on the real model.

## How bias is checked

1. **Slices.** Every out-of-fold prediction is grouped by the slices in `configs/bias.yaml`: time of day, rain, temperature, slope, precinct, era, volume tier, amenity tier, missing hours, public holidays, AFL days and their crosstabs. Each prediction comes from a model that never saw that sensor or its neighbours.
2. **Metric.** WAPE per group with Fairlearn's `MetricFrame`, with a 95% interval from resampling sensors (300 resamples). A group is **worse** when its whole interval is above the overall WAPE. Its **ratio** is its WAPE over the overall WAPE. The **worst slice** is the largest ratio among groups with at least 5 sensors.
3. **Selection after the bias check** (`src/models/flow/select.py`). Candidates within 3% of the best log MAE are compared on their worst slice, and the fairest wins. On 12 November that chose LightGBM (1.90) over XGBoost (2.24) for the rich variant.
4. **Mitigation** (`src/models/flow/mitigate.py`). The selected model is retrained on the same folds with each weighting of the training rows, or calibrated, and scored on the same slices and the same hours (those after each sensor's first 28 days).
5. **Gate** (`src/models/flow/registry.py`). A selected model whose worst slice is over 2.25 times the overall WAPE isn't registered. In `stoa_train`, that fails the `check_gates` task, which emails the team.

## What the slices show

For the selected rich model, six groups are clearly worse than overall (WAPE 0.645):

| Group | WAPE | Ratio | Sensors |
|---|---|---|---|
| Quiet sensors | 1.17 | 1.91 | 29 |
| Southbank | 0.82 | 1.33 | 6 |
| Melbourne (Remainder) | 0.76 | 1.23 | 6 |
| Low amenity | 0.74 | 1.20 | 33 |
| COVID era | 0.67 | 1.09 | 62 |
| Carlton | 0.77 | 1.26 | 1 (too thin for the worst-slice ratio) |

The portable model has the same pattern. Its worst groups are West Melbourne (Residential) at 1.95 times, on 4 sensors (under the 5-sensor minimum for the worst-slice ratio), and quiet sensors at 1.94 times, on 29.

Quiet sensors aren't missed by a bigger factor than busy ones. The median miss in a sensor's overall volume is 1.82 times either way at quiet sensors and 1.89 times at busy ones. But WAPE divides by counts, and hour-to-hour noise is a larger share of a small count (log MAE 1.05 at quiet sensors against 0.84 at busy ones). Most of their error is in the overall level, not the daily pattern.

## Mitigations tried

Rich variant (portable in brackets where it differs):

| Mitigation | log MAE | WAPE | Worst slice | Groups worse | Quiet | Night | COVID |
|---|---|---|---|---|---|---|---|
| None (selected model) | 0.864 | 0.645 | 1.91, quiet | 6 (10) | 1.17 | 0.68 | 0.67 |
| Sensor-equal weights | 0.888 | 0.628 | 2.06, quiet (2.18) | 8 (9) | 1.21 | 0.68 | 0.69 |
| Volume tiers weighted equally | 0.876 | 0.647 | 1.97, quiet | 5 (8) | 1.24 | 0.68 | 0.67 |
| Calibrated to 28 days of local counts | 0.595 | 0.405 | 1.82, night | 14 (15) | 0.57 | 0.73 | 0.63 |

- **Reweighting doesn't help.** Weighting sensors equally lowers overall WAPE a little, by fitting busy sensors better, but makes quiet sensors and the worst slice worse. Weighting the tiers equally does almost nothing. The data-stage study found the same with the stand-in model. Weights shift which rows the trees fit, but the quiet sensors' problem is their level, which no row weighting fixes.
- **Calibration does help.** Scaling each sensor to its first 28 days of counts halves the quiet sensors' error (1.17 to 0.57) and cuts the overall WAPE from 0.645 to 0.405.
- **What's left after calibration is a different problem.** More groups now count as "worse", but only because the overall error they're compared with fell so far; most of them improved in absolute terms. The new worst slice is night (0.73, 1.82 times), which calibration makes slightly worse: one scale per sensor can't fix a pattern that's wrong at some hours. Night, the COVID era, and Docklands need changes to the model's features, not weights or scales.

## Decision and trade-offs

- **Keep the selected model unweighted.** Both weightings cost accuracy and didn't make the quiet sensors fairer.
- **Calibration is the mitigation.** That's the product's design anyway: a site with even a few weeks of local counts gets calibrated predictions (CLAUDE.md, "Generalization"). The trade-off is that it needs local counts. A site without any keeps the uncalibrated model's errors.
- **For uncalibrated sites, say so in the app.** Predictions on quiet streets can be off by more than their own volume. Relative comparisons ("busier than", rankings, changes under a scenario) hold up better than absolute counts there. The intervals already widen for this.
- **The gate limit is 2.25.** It's set just under the XGBoost candidate's 2.24 and above today's 1.90 and 1.94, so a retrain that makes the worst slice noticeably worse is blocked and emailed, without failing on today's model. It's a team setting in `configs/models.yaml` (`gates.max_worst_slice_ratio`).
- **Next, for night and COVID:** features that describe night-time activity (venues open late, lighting), and a check on whether the COVID era should be down-weighted or dropped. Training from later start years (2014, 2020 or 2022) is on the list of reruns.

## Running it

```
.venv/bin/python -m src.models.flow.select      # selection, after the bias check
.venv/bin/python -m src.models.flow.mitigate    # the mitigations above, about 20 minutes
.venv/bin/python -m src.models.flow.registry --check
```

`mitigate` writes `data/models/<variant>/mitigation/summary.csv`, `slices.csv` (every group under every mitigation) and `mitigation.png`. Each mitigation is an MLflow run, `mitigation_<name>_<variant>`, with `stoa.role = mitigation`. The mitigations are listed under `flow.mitigation` in `configs/models.yaml`. `stoa_train` runs the step between selection and the gate.
