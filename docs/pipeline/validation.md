# Data validation

The `validate` task checks the hourly counts table (`data/processed/features/melbourne/counts_hourly.parquet`) after `join_all` builds it and before `dvc_push` versions it. It runs `src/pipeline/validate.py`, which uses [Great Expectations](https://greatexpectations.io) for the column checks. Every threshold is in `configs/validation.yaml`.

Great Expectations is pinned below 2.0 (`great-expectations>=1.9,<2` in `pyproject.toml`), since 2.0 changes its API.

```bash
.venv/bin/python -m src.pipeline.validate                 # run every check (41–47 s on 7.4M rows in Docker)
.venv/bin/python -m src.pipeline.validate --schema-diff   # how the table differs from the committed schema
.venv/bin/python -m src.features.schema --write           # regenerate the schema from src/features/schema.py
```

## Errors and warnings

Every check has a severity:

- **Error:** the data is wrong. The task fails without retrying (bad data fails the same way every time), Airflow sends the failure email, and `dvc_push` doesn't run, so a bad table never becomes a DVC version.
- **Warning:** worth a look, but not wrong. A big event or a sensor fault isn't a reason to stop the pipeline. Warnings go to the task log (`WARN` lines) and to the report, and **are emailed to the team** (`STOA_ALERT_EMAILS`, through the same `smtp_default` connection as the failure emails) with the run and each warning. The run carries on and the data is still versioned. A failed send is logged, not raised. A run with errors sends only the failure email.

Outliers and stuck sensors are only warned about in the **new hours**: those after the previous good run's newest hour (the last 7 days if there is no previous run). Older hits stay in the report as `earlier_rows` / `earlier_runs`, so a known event doesn't email the team every day. The other warnings already describe the current run (the last week's gaps, the newest hour, the change since the last run).

## The checks

On the run of 26 October 2026 (`data_snapshot_2026_10_26`, the committed data version): 71 checks, 0 errors, 2 warnings. The four per-minute checks were added after the run of 22 October (67 checks, 0 errors, 3 warnings). With the drift check against the training reference added, the run of 25 October on the merged pipeline had 73 checks, 0 errors and 1 warning (sensor 37 at 87% of last week's daytime hours).

| Kind | Severity | What it checks |
|---|---|---|
| **Schema** | error | The columns are exactly those in `configs/schemas/counts_hourly.yaml`, with the same pandas types. Catches a dropped, renamed, added or retyped column. |
| **Keys** | error | (`location_id`, `ts_utc`) is unique: one row per sensor-hour. |
| **Never missing** | error | 16 columns (`never_missing` in the config) have no empty values: keys, count, time and calendar flags. |
| **Missing only where documented** | error | Columns that may be empty only in the cases `docs/features/counts_hourly.md` describes. For example `segment_id` and the slope columns must be present when `has_location` is true (only sensors with no location may lack them; since the 18 retired sensors were placed from the 2022 sensor table, none do), the university calendar from 2015 on, and the directional counts for live rows. Each rule in `required_when` is `[column, operator, value]`, built into a Great Expectations `Column(...)` condition (text conditions are deprecated and removed in GX 2.0). |
| **Ranges** | error | 19 columns stay in a physical range: counts 0 to 30,000 (the record hour is about 16,000), hours 0–23, humidity and cloud 0–100%, slope 0–40%, and so on. |
| **Allowed values** | error | `source` is `archive`, `live` or `wayback`; `place_source` is `footpath_steepness` or `ga_dem_5m`. |
| **Weather present** | warning | Each `wx_*` column is present for at least 99% of rows. The weather archive lags about a week, so the newest rows have none until a later run fills them. |
| **Outliers** | warning | New hours above 4x their own sensor's 99.9th percentile (from its whole history). Most big events stay below that; past hits were real events (below). |
| **Stuck sensors** | warning | A run of the same count (10 or more) for 6+ hours at one sensor, reaching into the new hours. Small counts repeat naturally overnight (1 an hour), so they're ignored. |
| **Outage share** | warning | At most 1% of rows flagged `suspect_outage_zero`. |
| **Recent completeness** | warning | Each sensor reported at least 90% of the last 7 days' **daytime** hours (08:00–20:00), and no sensor that reported in the 30 days before has gone silent. Night hours aren't expected: the live portal leaves out hours with no one counted. |
| **Freshness** | warning | The newest hour is at most 5 days old. |
| **Drift: rows** | error | The table didn't lose more than 1% of its rows since the last good run. Held rows are kept and new hours added, so the row count only grows. |
| **Drift: sensors, missing values, mean** | warning | No fewer sensors than last time; no column's missing share up by more than 5 points; mean hourly count within 20% of last time. |
| **Drift: against the training reference** | warning | Each feature's distribution in the rows a model may use, against the reference the training frame stored (`docs/features/training_frame.md`): a population stability index above 0.25 warns. By default the reference is the latest frame, so this catches a change in how a feature is built; pointed at a deployed model's frame (`drift.training_reference`), it measures drift since that model was trained. |
| **Per-minute archive: days** | warning | Every Melbourne day since the archive began has a file in `data/raw/melbourne_portal/minute_feed/`. With no archive at all (a clone without bucket access), one passing check says so. |
| **Per-minute archive: late polls** | warning | Each finished day's last minute is 23:00 or later. The feed keeps a day only until a reset at about 02:00 the next morning, so an earlier end means no poll landed between midnight and the reset. |
| **Per-minute archive: duplicates** | warning | At most 50 rows share a sensor and minute. The portal publishes a few (10 on 26 October); the archive keeps them as published. |
| **Per-minute archive: hourly sums** | warning | For hours both tables cover, each sensor-hour's minutes add up to its hourly count within 5% (or 2 people), for at least 95% of sensor-hours. A match also shows the absent minutes were zeros. The hourly table's newest hour is left out (the portal publishes it unfinished), and so is the archive's first hour when the archive began mid-hour. |

The per-minute checks are all warnings: the archive is extra evidence for within-hour profiles, not the training table. Their settings are under `minute_feed` in `configs/validation.yaml`; the code is `src/pipeline/minute_checks.py`.

Each rule was checked against the real table before it was set: on clean data none of the error checks fail, and every rule that failed on a test copy with planted errors (negative counts, duplicate sensor-hours, a dropped column, a retyped column, an unknown `source`, a located sensor with no segment, 3% of rows removed) was reported as an error.

The full DAG on a fresh clone (table rebuilt from public sources, 6,176,935 rows): 64 checks, 0 errors, 2 warnings. Warnings from a run by hand on 22 October 2026, on a table built on 16 October:
- Sensors 37, 164 and 167 reported 87–88% of the last week's daytime hours.
- The newest hour was 6.3 days old (the local copy was a week old; a scheduled run pulls the latest counts first).

On 26 October the hourly-sums check warned that 28.9% of 253 sensor-hours didn't add up. Every one of them is the archive's first hour, 00:00 on 25 October: the archive began at 00:55 that night, so it holds only that hour's last 5 minutes (minute sums a median 116 below the hourly count). The two full hours after it matched for every sensor. Only three hours overlap so far, because the hourly table trails the per-minute feed, so this one partial hour weighs heavily; its share falls as hours arrive. Since 25 October both the check and `minute_profiles` leave out the archive's first hour when the archive began mid-hour, so this warning doesn't recur; a gap later in the archive still warns.

The outlier check also finds 9 older hours, at sensors 8, 10, 71 and 75. They are real crowds, not glitches: sensor 8 at 7 a.m. on four July Sundays (2019, 2022, 2025, 2026) is Run Melbourne, sensor 10 at midnight on 1 January 2025 is New Year's Eve, and sensor 71 is Christmas morning. That is why only new hours warn.

## Schema and statistics over time

- **Schema:** `configs/schemas/counts_hourly.yaml`, generated from `src/features/schema.py`, which declares every column's type, role (key, label, feature, aux or meta) and whether it is portable. When the table's columns change on purpose (a new feature), declare them in `schema.py`, regenerate the YAML and commit both; `tests/features/test_schema.py` fails if they disagree. `--schema-diff` lists what a rebuilt table has that the schema doesn't.
- **Statistics:** every run profiles each column (type, missing share; min, max, mean, standard deviation and median for numbers; true share for flags; distinct values and the most common ones for text) and the table (rows, sensors, first and last hour, mean count).
- **Over time:** each good run appends its summary to a history file, and the next run's drift checks compare against the last line. Failed runs don't join the history, so a bad table never becomes the baseline. The reports are written under `data/processed/`, so DVC versions them with the data they describe.

| File (in `data/processed/validation/melbourne/`) | Contents |
|---|---|
| `counts_hourly_validation.json` | Every check of the latest run: name, severity, passed, details |
| `counts_hourly_stats.json` | The latest statistics profile |
| `counts_hourly_stats_history.jsonl` | One summary line per good run, oldest first |

## Speed

Great Expectations makes one pass over the 6.2M rows per check. Running the suite with `result_format="BOOLEAN_ONLY"` and re-running only failed checks with `SUMMARY` (for counts and example rows) cut it from 73 s to 33 s. The whole task takes about 40 s run directly on a Mac and about 85 s in the Airflow container (Docker Desktop) when it runs alone; beside `bias_slices`, which also reads the whole table, it took 218 s on 23 October 2026, and once the table reached 7.4M rows the pair ran out of Docker's 8 GB, so `bias_slices` now runs after it (peaks measured natively: `validate` 3.3 GB, `bias_slices` 2.9 GB). That makes it the slowest task of a daily run (README, "Pipeline optimization"). Possible next steps if that matters: validate the fixed 2009–2022 archive rows once and only the live rows each day, or give the container more CPUs.

## Changing a check

Edit `configs/validation.yaml` and run the module by hand to see the effect. A new column in the table needs an entry in `src/features/schema.py` and a regenerated schema, plus a range or `required_when` rule if it has one. Tests with small made-up tables are in `tests/pipeline/test_validate.py`.
