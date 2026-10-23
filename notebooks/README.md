# Melbourne dataset notebooks

These notebooks are for exploration and demonstration. The `src/` pipeline, run by the Airflow DAG, is what builds the project's data; every step worked out here now lives there (the README's "Notebooks are not the pipeline" table says where).

Notebooks 01-03 build a Melbourne hourly pedestrian dataset with footpath steepness, runnable in Colab or locally; 04 and 05 are the bias checks. Run them in order:

| Step | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `01_download.ipynb` | the web (the only notebook that goes online) | `raw/` |
| 2 | `02_combine.ipynb` | `raw/` | `interim/`, `processed/` |
| 3 | `03_preprocess.ipynb` | `raw/`, `interim/`, `processed/` | `final/` |
| 4 | `04_bias_slicing.ipynb` | the team's bias report, `data/processed/bias/melbourne/` (from `src.bias.representation` or `dvc pull`), and `configs/bias.yaml` | the figures in `docs/bias/figures/` |
| 5 | `05_bias_stage1.ipynb` | `final/`, `interim/missing_hours.parquet` | `reports/bias/`: the same data-bias check on this dataset, which includes the Wayback fill |

Notebook 04 runs locally from the repo only, in the project environment (`.venv`); it doesn't use the data folder below. Notebooks 01-03 and 05 use the data folder.

Paths are relative to the data folder:
- **Colab:** `MyDrive/stoa/data/` on Google Drive (mounted by the first cell).
- **Local clone:** `data/notebooks/` in the repo (git-ignored), kept apart from the `src/` pipeline's `data/raw`, `data/interim` and `data/processed`. Set `STOA_NOTEBOOK_DATA` to use another folder. Run the notebooks from `notebooks/`.

The data isn't in DVC yet.

## The two final datasets

| File | One row per | Key | Contents |
|---|---|---|---|
| `final/pcs_hourly.parquet` | sensor-hour (~7.4M rows, ~22 MB) | `location_id` + `date` + `time` | `latitude`, `longitude`, UTC `date` and `time`, `pedestriancount`, `direction_1`, `direction_2` (NaN before July 2021), `source`, `suspect_outage_zero`, `temp_c`, `wind_kmh`, `is_day`, `rain_mm` |
| `final/sensor_steepness.parquet` | sensor (118 rows) | `location_id` | location, `slope_pct`, `elevation_m`, direction labels, how each value was matched (`match_distance_m`, `slope_matched`, `place_source`) |

Join them on `location_id`. Coverage: 2009-05-01 (UTC) to the last hour with weather, about 6 days before the download (ERA5 lag), so every row has weather. Row counts grow with each new download.

`direction_1` and `direction_2` add up to `pedestriancount`, so never use them as features.

## Sources (`01_download.ipynb`, in download order)

| # | Source | Coverage |
|---|---|---|
| 1 | Counts archive (portal attachment) | 2009-05-01 to 2022-10-31 |
| 2 | Counts, Wayback Machine copy of the portal export (snapshot 2025-05-02) | 2021-07-01 to 2025-05-02 |
| 3 | Counts, live portal table (dated file per pull; rolling ~2 years) | 2024-10-21 to now |
| 4 | Current sensor locations | snapshot |
| 5 | Sensor locations, Oct 2022 Wayback copy (18 retired sensors) | snapshot |
| 6 | Footpath steepness (City of Melbourne) | snapshot |
| 7 | GA 5 m LiDAR ground heights, 5 points around each sensor | static |
| 8 | ERA5 weather via Open-Meteo: `temperature_2m`, `precipitation`, `wind_speed_10m`, `is_day` | 2009-04-30 to ~6 days ago |

All CC BY 4.0 (City of Melbourne, Open-Meteo/ERA5, Geoscience Australia).

## How this relates to the `src/` pipeline

`src/features/counts_hourly.py` builds a similar hourly table. These notebooks first added three things, and all three are now in the pipeline:

1. **The Nov 2022 to Oct 2024 gap.** `counts_hourly` uses the archive and the live table, so it has no rows between the archive's end and the live window's start (6.17M rows vs 7.42M here). The Wayback copy fills that gap. Where sources overlap, live wins, then the archive, then Wayback (the Wayback copy matched the others on ~98.5% of overlapping hours, per the check noted in `01_download`).
2. **Locations for the 18 archive sensors** missing from the current sensor-locations file (an open decision in `CLAUDE.md`): taken from the Oct 2022 copy of the same City of Melbourne table.
3. **Footpath slope and elevation per sensor** from the footpath-steepness survey, with a LiDAR fallback for the two sensors (25 and 130) with no footpath section within 25 m.

Where they are now: the Wayback fill in `counts_hourly.fill_gap` (PR #16), the 18 sensors' locations in `sensor_match.load_sensor_locations`, and the slope in `src/features/sensor_steepness.py`. The September 2010 fix below is `counts_hourly.rehour_midnight_days`. The pipeline differs in two ways: the Wayback copy is used only for the gap, so archive and live rows stay as published, and its outage rule is the one described in `docs/features/counts_hourly.md`.

## Processing notes

- **Sep 2010 archive bug:** every row that month is stamped hour 0; hours are recovered from the row order (12,240 rows). Done in `02_combine` because the source merge would otherwise drop them as duplicates.
- **Daylight saving:** counts are local clock time; they're converted to UTC (the skipped October 02:00 is dropped, the repeated April 02:00 is read as daylight time), and outputs are UTC only.
- **Footpath matching:** sections with no grade (4,455) or shorter than 5 m (4,779; their grades reach 580%) are skipped. `slope_pct` is the footpath file's `gradepc`: metres up per 100 m. Sensor 139 matches a 20.4% section, likely stairs or a ramp.
- **Outage zeros:** runs of 24+ consecutive zero hours at one sensor are flagged `suspect_outage_zero` (3,663 rows), not deleted. `counts_hourly` uses a different rule (3+ hours touching 08:00 to 20:00).
- **Indoor sensors:** dropped in `03_preprocess`; none of the 34 indoor sensors appear in the count files.
- **Direction labels:** for the 31 North/South sensors in both sensor tables, the 2022 copy says direction 1 = South and the current table says North; East/West labels agree. Check before relying on the labels.
