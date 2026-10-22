# Data validation

The `validate` task checks the hourly counts table (`data/processed/features/melbourne/counts_hourly.parquet`) after `join_all` builds it and before `dvc_push` versions it. It runs `src/pipeline/validate.py`, which uses [Great Expectations](https://greatexpectations.io) for the column checks. Every threshold is in `configs/validation.yaml`.

```bash
.venv/bin/python -m src.pipeline.validate                 # run every check (~40 s on 6.2M rows)
.venv/bin/python -m src.pipeline.validate --write-schema  # regenerate the schema from the table
```

## Errors and warnings

Every check has a severity:

- **Error:** the data is wrong. The task fails without retrying (bad data fails the same way every time), Airflow sends the failure email, and `dvc_push` doesn't run, so a bad table never becomes a DVC version.
- **Warning:** worth a look, but not wrong. A big event or a sensor fault isn't a reason to stop the pipeline. Warnings go to the task log (`WARN` lines) and to the report; the run carries on.

## The checks

On the run of 22 October 2026: 67 checks, 0 errors, 3 warnings.

| Kind | Severity | What it checks |
|---|---|---|
| **Schema** | error | The columns are exactly those in `configs/schemas/counts_hourly.yaml`, with the same pandas types. Catches a dropped, renamed, added or retyped column. |
| **Keys** | error | (`location_id`, `ts_utc`) is unique: one row per sensor-hour. |
| **Never missing** | error | 16 columns (`never_missing` in the config) have no empty values: keys, count, time and calendar flags. |
| **Missing only where documented** | error | Columns that may be empty only in the cases `docs/features/counts_hourly.md` describes. For example `segment_id` and the slope columns must be present when `has_location` is true (only the 18 sensors with no location lack them), the university calendar from 2015 on, and the directional counts for live rows. Each rule is a pandas query in `required_when`. |
| **Ranges** | error | 19 columns stay in a physical range: counts 0 to 30,000 (the record hour is about 16,000), hours 0–23, humidity and cloud 0–100%, slope 0–40%, and so on. |
| **Allowed values** | error | `source` is `archive` or `live`; `place_source` is `footpath_steepness` or `ga_dem_5m`. |
| **Weather present** | warning | Each `wx_*` column is present for at least 99% of rows. The weather archive lags about a week, so the newest rows have none until a later run fills them. |
| **Outliers** | warning | Hours above 4x their own sensor's 99.9th percentile. Big events reach 2–3x; far beyond that is usually a glitch. |
| **Stuck sensors** | warning | The same count (10 or more) for 6+ hours in a row at one sensor. Small counts repeat naturally overnight (1 an hour), so they're ignored. |
| **Outage share** | warning | At most 1% of rows flagged `suspect_outage_zero`. |
| **Recent completeness** | warning | Each sensor reported at least 90% of the last 7 days' **daytime** hours (08:00–20:00), and no sensor that reported in the 30 days before has gone silent. Night hours aren't expected: the live portal leaves out hours with no one counted. |
| **Freshness** | warning | The newest hour is at most 5 days old. |
| **Drift: rows** | error | The table didn't lose more than 1% of its rows since the last good run. Held rows are kept and new hours added, so the row count only grows. |
| **Drift: sensors, missing values, mean** | warning | No fewer sensors than last time; no column's missing share up by more than 5 points; mean hourly count within 20% of last time. |

Each rule was checked against the real table before it was set: on clean data none of the error checks fail, and every rule that failed on a test copy with planted errors (negative counts, duplicate sensor-hours, a dropped column, a retyped column, an unknown `source`, a located sensor with no segment, 3% of rows removed) was reported as an error.

The full DAG on a fresh clone (table rebuilt from public sources, 6,176,935 rows): 64 checks, 0 errors, 2 warnings (the 9 outlier hours below, and sensor 37 at 85% of daytime hours). Warnings from a run by hand on 22 October 2026, on a table built on 16 October:
- 9 hours above 4x their sensor's 99.9th percentile, at sensors 8, 10, 71 and 75.
- Sensors 37, 164 and 167 reported 87–88% of the last week's daytime hours.
- The newest hour was 6.3 days old (the local copy was a week old; a scheduled run pulls the latest counts first).

## Schema and statistics over time

- **Schema:** `configs/schemas/counts_hourly.yaml`, generated from the table with `--write-schema` and committed. When the table's columns change on purpose (a new feature), regenerate it and commit it with that change; the diff shows what changed.
- **Statistics:** every run profiles each column (type, missing share; min, max, mean, standard deviation and median for numbers; true share for flags; distinct values and the most common ones for text) and the table (rows, sensors, first and last hour, mean count).
- **Over time:** each good run appends its summary to a history file, and the next run's drift checks compare against the last line. Failed runs don't join the history, so a bad table never becomes the baseline. The reports are written under `data/processed/`, so DVC versions them with the data they describe.

| File (in `data/processed/validation/melbourne/`) | Contents |
|---|---|
| `counts_hourly_validation.json` | Every check of the latest run: name, severity, passed, details |
| `counts_hourly_stats.json` | The latest statistics profile |
| `counts_hourly_stats_history.jsonl` | One summary line per good run, oldest first |

## Speed

Great Expectations makes one pass over the 6.2M rows per check. Running the suite with `result_format="BOOLEAN_ONLY"` and re-running only failed checks with `SUMMARY` (for counts and example rows) cut it from 73 s to 33 s. The whole task takes about 40 s run directly on a Mac and about 85 s in the Airflow container (Docker Desktop), which makes it the slowest task of a daily run. Possible next steps if that matters: validate the fixed 2009–2022 archive rows once and only the live rows each day, or give the container more CPUs.

## Changing a check

Edit `configs/validation.yaml` and run the module by hand to see the effect. A new column in the table needs `--write-schema`, plus a range or `required_when` rule if it has one. Tests with small made-up tables are in `tests/pipeline/test_validate.py`.
