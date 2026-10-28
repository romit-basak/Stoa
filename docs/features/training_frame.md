# Training frame

`src/features/training_frame.py` decides what any model trained on the counts sees: which rows, which columns, and which sensors it may train on. It runs as the `training_frame` task after `validate`, and the bias baseline (`src/bias/slice_errors.py`) builds its inputs with it, so every model shares one set of tested leak traps.

```bash
.venv/bin/python -m src.features.training_frame                      # ~10 s
.venv/bin/python -m src.features.training_frame --drop-needs-review  # sensitivity run
```

## What it guards

| Rule | How |
|---|---|
| Only the training site trains (CLAUDE.md: NYC is a transfer test, never training data) | `check_site` rejects any site that isn't `counts_hourly.site` with `count_sensors` |
| Rows | Located outdoor sensors, hours not flagged `suspect_outage_zero` |
| No IDs, coordinates, target or its parts as features (core rule 9) | Columns come from `src/features/schema.py` by role: only `feature` columns enter X. `direction_1`/`direction_2` (they add up to the count), `hour_of_week` (weekday × hour) and `length_m` (depends on how a way was split) are `aux` |
| Portable model (CLAUDE.md, "Generalization") | `variant="portable"` keeps the features any city has (138 of 143); `rich` adds the Melbourne-only ones (`post_metro_tunnel`, the footpath-survey slope and elevation, `recent_station_dist_m`) |
| The OSM snapshot postdates the Metro Tunnel | Before 30 November 2025 the new stations' distance (and their entrances') is set to the cap, as if they weren't there |
| Split by location (core rule 2) | Sensors that are all within 200 m of each other share a group (complete linkage, so a dense street doesn't chain into one group); folds hold out whole groups |
| Within-Melbourne transfer test | The holdout precincts in `configs/bias.yaml` (UniMelb edge, Carlton, North Melbourne), and any sensor grouped with one, are in no fold (`fold = -1`) |
| Event days reported apart | `event_days.parquet`: dates with an AFL game at the MCG or Marvel, or a logged event at a venue (`configs/events.yaml`) |

The frame isn't written out: 7.4M rows by 143 features is about 4 GB as float32. `Frame.matrix()` builds it in chunks of 500,000 rows.

## Outputs (`data/processed/training/melbourne/`)

| File | Contents |
|---|---|
| `splits.parquet` | One row per sensor: `group`, `fold` (0–4, or -1 when held out), `holdout` (the precinct, `neighbour`, or empty) |
| `event_days.parquet` | `date_local`, the kinds of event and the venues |
| `frame_spec.json` | Row and sensor counts, the feature list of each variant, SHA-256 of the input tables, the DVC data version (`data/processed.dvc`), the OSM extract's SHA-256, the number of sensors whose match needs review, and a reference distribution of every feature |

A model run should log `frame_spec.json` (and its DVC and OSM versions) with its results.

The reference distributions are what `validate` compares the table against each day (`drift: against the training reference`, docs/pipeline/validation.md): ten bins at the deciles for a numeric feature, or each value's share for one with few values (flags, hours, weekdays).

## Run of 25 October 2026

7,399,986 rows from 118 sensors; 14 sensors held out (6 North Melbourne, 5 Carlton, 3 UniMelb edge); 58 sensor groups of up to 6 sensors; folds of 18 to 28 sensors; 1,239 event days. 52 sensors have a match that needs review (the hand check was skipped), so `--drop-needs-review` exists to check how much they matter.

The bias baseline's inputs through the frame were checked against the code they replaced: identical feature matrices (200,000 rows × 143 columns) and identical fold plans.
