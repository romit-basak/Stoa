# Training periods: does dropping older years help?

The Melbourne counts go back to 2009. Older years might teach the flow model patterns that no longer hold: the city changed, the COVID lockdowns broke the usual rhythms, and the Metro Tunnel opened on 2025-11-30. To test this we retrained the model from three later starts (2014, 2020 and 2022) and compared each with the current model, trained on everything from 2009.

**Result:** keep the 2009 start. Every later start is worse on all hours, and for the rich variant on the held-out precincts too. The 2022 start does a little better on the most recent hours: rich log MAE is 0.799 against 0.820 after the Metro Tunnel opened. It gets there by giving up everything older, though, and the gap is small enough that one seed can't settle it.

## How the runs were set up

- `train --train-start YYYY-01-01` (Melbourne time) limits only the **training** rows. Each fold still predicts the same out-of-fold hours, so every run is scored on exactly the same 6,456,502 hours. The location folds, the 600,000-row sample per fold and the seed are unchanged.
- `--params-from` reuses the 2009 run's tuned hyperparameters, so the data period is the only thing that changes. The later starts were not retuned.
- Selection then ran on each run as usual: family choice, interval models, holdout precincts, eras and slices.
- The runs are in the MLflow experiment `stoa-flow-periods` (set with `MLFLOW_EXPERIMENT_NAME`), so `registry --check` never compares them with the main `stoa-flow` runs.
- `python -m src.models.flow.periods` scores each run's selected model on all hours and on the hours from 2014, 2020 and 2022. All runs use the same data version (`data/processed.dvc` md5 `961075e6`).

Compare runs on log MAE directly, not on the gain against Space Syntax. The Space Syntax baseline is also fitted on the training rows, so it changes with the start date too. On all hours it scores 0.979 for the 2009 start but 1.022 for the 2020 start. The 2020 run looks almost as good as 2009 on that gain (11.6% against 11.7%), but only because its own baseline got worse.

## Results

Out-of-fold log MAE of the selected model (lower is better). Windows are the hours from the given date onwards.

**Rich variant**

| Training start | Family | All hours | From 2014 | From 2020 | From 2022 |
|---|---|---|---|---|---|
| 2009 (current) | lgbm | **0.864** | **0.866** | **0.873** | 0.854 |
| 2014 | lgbm | 0.880 | 0.884 | 0.891 | 0.876 |
| 2020 | xgb | 0.903 | 0.906 | 0.894 | 0.876 |
| 2022 | lgbm | 0.908 | 0.911 | 0.914 | **0.838** |

**Portable variant**

| Training start | Family | All hours | From 2014 | From 2020 | From 2022 |
|---|---|---|---|---|---|
| 2009 (current) | lgbm | **0.878** | **0.880** | **0.885** | 0.863 |
| 2014 | lgbm | 0.886 | 0.890 | 0.893 | 0.873 |
| 2020 | xgb | 0.930 | 0.932 | 0.915 | 0.897 |
| 2022 | lgbm | 0.927 | 0.930 | 0.935 | **0.859** |

By era (eras from `configs/bias.yaml`; scores from selection's era table, which averages differently from the fold means above):

| Era | Hours | Rich 2009 | Rich 2022 | Portable 2009 | Portable 2022 |
|---|---|---|---|---|---|
| Pre-COVID (to 2020-03-15) | 2.74 M | **0.808** | 0.866 | **0.832** | 0.894 |
| COVID (2020-03-16 to 2021-10-21) | 0.76 M | **0.913** | 1.206 | **0.949** | 1.228 |
| Recovery (to 2025-11-29) | 2.28 M | 0.847 | **0.836** | 0.855 | 0.855 |
| After the Metro Tunnel (from 2025-11-30) | 0.67 M | 0.820 | **0.799** | 0.839 | **0.833** |

Holdout precincts, uncalibrated log MAE (whole precincts never seen in training):

| Precinct | Rich 2009 | Rich 2014 | Rich 2020 | Rich 2022 |
|---|---|---|---|---|
| All precincts together | **0.874** | 0.891 | 0.912 | 0.909 |
| Carlton | 0.832 | 0.840 | **0.821** | 0.901 |
| North Melbourne | 0.844 | 0.827 | 0.851 | **0.809** |
| UniMelb edge, before the Metro Tunnel | **0.959** | 1.021 | 1.094 | 1.010 |
| UniMelb edge, after the Metro Tunnel | 1.019 | 1.068 | 1.220 | **0.970** |

The portable variant is the exception: on the precincts together the 2020 start scores 1.013, against 1.082 for 2009 (1.100 for 2014, 1.108 for 2022). Nearly all of that comes from North Melbourne (0.969 against 1.304); on Carlton and the UniMelb edge the 2020 start is no better than 2009. Interval coverage stays within 0.77 to 0.79 and the worst slice ratio within 1.78 to 2.01 for every run, so no start fails those checks.

## What this shows

- **More data wins on most hours.** Starting in 2014 already costs about 0.016 of log MAE on all hours (rich), and it is worse in every window, including the hours from 2014 that it was trained on. The older years help the model learn how place and time shape counts, and that carries over to later years.
- **Starting in 2020 doesn't help anywhere.** About a fifth of its training hours fall in the lockdowns, and it has none of the normal years before them. It is worse than the 2009 start even on the hours from 2020.
- **Starting in 2022 helps a little on recent hours only.** Rich log MAE after the Metro Tunnel drops from 0.820 to 0.799 (2.6% lower); portable drops from 0.839 to 0.833 (0.7%). The UniMelb edge after the Metro Tunnel also improves (1.019 to 0.970). The cost is large on everything older: COVID-era error goes from 0.913 to 1.206, which is expected, because the 2022 model never saw a lockdown.
- **None of this moves the 15% gate.** The gate scores all hours, where the 2009 start is best.

## Caveats

- One seed per run and no retuning per period. A 0.02 difference on the most recent era could be partly noise, and a model tuned for the 2022 rows might do better or worse.
- The 2022 start trains on fewer distinct hours, but the per-fold sample is still 600,000 rows. The comparison is between periods of data, not amounts of training.
- Sensors that stopped before 2022 have no training rows in the 2022 run, but their hours are still scored. That is part of the cost of a late start, not an error in the comparison.

## If we want the recent-hours gain

Cutting the old years is a blunt way to get it. A better next step is to keep all years and weight recent rows more heavily in training, which could keep the 2009 model's breadth and pick up some of the recent gain. That would be a new experiment run the same way: same folds, same scored hours, its own MLflow experiment. It's optional and not needed for Tier 1. The champion stays on the 2009 start unless the team decides otherwise.

## Reproduce

```bash
export MLFLOW_EXPERIMENT_NAME=stoa-flow-periods
for y in 2014 2020 2022; do
  .venv/bin/python -m src.models.flow.train --train-start $y-01-01 \
      --params-from data/models --output-dir data/models/periods/$y
  .venv/bin/python -m src.models.flow.select --output-dir data/models/periods/$y
done
.venv/bin/python -m src.models.flow.periods --run 2009=data/models \
    --run 2014=data/models/periods/2014 --run 2020=data/models/periods/2020 \
    --run 2022=data/models/periods/2022 --windows 2014-01-01 2020-01-01 2022-01-01 \
    --output data/models/periods/periods.csv
```

Each train-plus-select pair takes about 40 to 50 minutes on a MacBook.
