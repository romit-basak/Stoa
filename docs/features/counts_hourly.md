# Hourly counts table (Melbourne)

`data/processed/features/melbourne/counts_hourly.parquet` holds one row per (sensor, hour): the pedestrian count with its time, calendar, weather and sensor-match columns. It's the base table for training. Build it with:

```bash
.venv/bin/python -m src.features.counts_hourly    # ~30 s, ~2.5 GB memory
```

It runs as `join_all` in the Airflow DAG. It needs the counts, weather and calendars pulled, `sensor_osm_features.parquet` from the OSM pipeline, and `sensor_steepness.parquet` (below). Settings are in `configs/features.yaml` under `counts_hourly`. Run statistics go to `counts_hourly_meta.json`.

## Adding the segment features

The ~115 OSM segment columns aren't copied onto every row (about 9 GB in memory). Join them for the rows you train on:

```python
import pandas as pd

hourly = pd.read_parquet("data/processed/features/melbourne/counts_hourly.parquet")
hourly = hourly[hourly.has_location & (hourly.location_type == "Outdoor") & ~hourly.suspect_outage_zero]
seg = pd.read_parquet("data/processed/features/melbourne/sensor_osm_features.parquet")
seg = seg.drop(columns=["sensor_description", "location_type", "segment_id", "match_method", "needs_review"])
df = hourly.merge(seg, on="location_id", how="left", validate="many_to_one")

hours = pd.read_parquet("data/processed/features/melbourne/osm_segment_open_hours.parquet")
df = df.merge(hours, on=["segment_id", "hour_of_week"], how="left")   # missing rows mean 0
```

Before 2025-11-30, set `recent_station_dist_m` to its cap (`post_metro_tunnel` is False for those rows).

## Count sources

| Source | File in `data/raw/melbourne_portal/` | Holds | Used for |
|---|---|---|---|
| `archive` | `counts_per_hour_2009-05_to_2022-12-14.csv.zip` (portal attachment) | 2009-05-01 to 2022-10-31, despite the file name | all of it |
| `live` | `pedestrian-counting-system-monthly-counts-per-hour.parquet` | a rolling ~2-year window, merged into the rows held | all of it |
| `wayback` | `counts_per_hour_wayback_2025-05-02.csv` | the portal's full CSV export as the Internet Archive saved it on 2025-05-02: 2021-07-01 to 2025-05-02 | only hours after the archive's last hour and before the live table's first hour |

The live window had already moved past November 2022 to October 2024 by the time the pipeline started, so those months exist only in the Wayback copy (configured under `melbourne_portal.wayback` in `configs/data.yaml`; `melbourne_portal` downloads it once and keeps it). It's the City's own export, not a different collection: where it overlaps the archive (668,180 sensor-hours) and the live table (427,607), it has the same count on 98.5% of hours. Where it differs, the median difference is 24 people against the archive and 150 against the live table, so it isn't used where either of those has the hour. Taking it only for the gap gives every hour one source and leaves the archive and live rows exactly as published.

## Time handling

People follow the local clock, so **hour, weekday and every calendar column use Melbourne local time, daylight saving included.** Physical quantities use the real instant: weather is joined on UTC, and sun elevation is computed from UTC.

The sources don't agree on clocks, so this matters:

| Source | Clock | How we know |
|---|---|---|
| Counts, archive (2009–2022) | Local, with daylight saving | The weekday morning peak is at 8 a.m. in both January and July |
| Counts, live (2024–now) | Local, with daylight saving | The October changeover day has 23 hours |
| Counts, Wayback copy (2021–2025; used for 2022–2024) | Local, with daylight saving | The portal's own export, the same table as live |
| Weather (Open-Meteo) | **Fixed UTC+10 all year** | Each file has one `utc_offset_seconds`, 8,760 hours a year, and radiation peaks at the same clock hour in January and July |

Joining weather on the local clock would put it an hour out from October to April every year. Joining on UTC fixes that.

Within each count hour (h:00 to h+1:00):
- **Instantaneous weather** (temperature, humidity, cloud, wind, `is_day`) comes from the value stamped at h.
- **Preceding-hour values** (`precipitation`, `rain`, `shortwave_radiation`) come from the value stamped at h+1, so they cover the same hour as the count.
- **Sun elevation** is taken at the middle of the hour. That's why `wx_is_day` (sampled at h) and `sun_elevation_deg > 0` disagree on about 4% of rows, all around sunrise and sunset.

**Limitations at the clock changes** (once each way per year, at about 2 a.m.):
- **October:** the clocks skip 02:00. The archive still has rows stamped with that hour, and they're dropped (370 rows).
- **April:** 02:00 happens twice, but the sources have one row for it. We read it as the first (daylight-saving) occurrence, so the other hour of the pair is missing.

## Columns

| Column | Meaning |
|---|---|
| `location_id` | Sensor (same ID as the archive's `Sensor_ID`). Never a model feature (core rule 9). |
| `ts_utc`, `ts_local` | Start of the hour, in UTC and in Melbourne time |
| `date_local`, `hour_local`, `weekday` (Monday = 0), `hour_of_week`, `is_weekend` | Local-clock time features. `hour_of_week` joins the opening-hours table. |
| `source` | `archive` (2009-05 to 2022-10), `wayback` (the gap between them, 2022-11 to 2024-10) or `live` (2024-10 onward). See "Count sources" below. |
| `pedestriancount` | Count for the hour (the target) |
| `direction_1`, `direction_2` | Directional counts: live and Wayback rows only, NaN in the archive. They add up to `pedestriancount`, so **never use them as features**: they would leak the target. |
| `suspect_outage_zero` | In a run of ≥ 3 consecutive zero hours that touches 08:00–20:00. Likely a sensor fault, not an empty street. Drop these rows when training. |
| `is_public_holiday` | Victorian public holiday (`holidays` package) |
| `school_term` | Within a Victorian school term (student days; weekends included). Missing outside 2009–2026. |
| `uni_teaching`, `uni_exams` | University of Melbourne teaching weeks (breaks excluded) and exam periods. **Missing before 2015** (no calendar), not False. |
| `sun_elevation_deg` | Sun above the horizon, in degrees, at the middle of the hour (NOAA formula, city centre) |
| `covid_period` | 16 March 2020 to 21 October 2021: flagged, not dropped |
| `post_metro_tunnel` | On or after 30 November 2025 (structural break) |
| `wx_*` | Open-Meteo ERA5 for the city centre: `temperature_2m`, `apparent_temperature`, `relative_humidity_2m`, `cloud_cover`, `wind_speed_10m`, `is_day`, `precipitation`, `rain`, `shortwave_radiation` |
| `sensor_description`, `location_type`, `segment_id`, `match_method`, `needs_review` | From the sensor match (`sensor_segment_map.parquet`) |
| `has_location` | The sensor has a location: in the current sensor-locations file, or for the 18 retired sensors, its October 2022 copy |
| `slope_pct`, `slope_pct_local_mean`, `elevation_m`, `place_source` | Footpath slope (%), the length-weighted mean slope within 25 m, and elevation (m AHD), from `sensor_steepness.parquet` (below). Missing for sensors with no location. |

There's no month or day-of-year column, by design (core rule 9): use `sun_elevation_deg`, temperature and the calendar flags instead.

## Footpath slope per sensor

`src/features/sensor_steepness.py` (task `sensor_steepness`) writes `sensor_steepness.parquet`, one row per outdoor sensor. It's based on Shubham Hadawle's Colab join notebook. Each sensor is matched to the nearest section of the City of Melbourne footpath-steepness survey within 25 m:
- `slope_pct` is that section's `gradepc` (metres of rise per 100 m along the footpath).
- `elevation_m` is the mean of its `rlmin` and `rlmax`.
- `slope_pct_local_mean` is the length-weighted mean `gradepc` of all sections within 25 m, so one odd section doesn't decide the value.

Sections with no grade (4,455) or shorter than 5 m (4,854; their grades reach 580%) are skipped. Skipping the ungraded ones matters: without it, sensor 143 was matched to an empty section and got no slope.

Sensors 25 and 130 have no section within 25 m. Their elevation and slope come from the GA 5 m LiDAR DEM, with slope taken from samples 10 m to each side. That's the steepest terrain gradient, not the grade along a footpath, so `place_source` (`footpath_steepness` or `ga_dem_5m`) records which method each sensor used.

Sensor 139 (Sandridge Bridge signal box, by a ramp) has 20.4%; every other sensor is under 10%. The footpath survey exists only for the City of Melbourne, so these columns belong to the Melbourne-rich model, not the portable one (CLAUDE.md, "Generalization").

## Run of 22 October 2026

With the Wayback fill, September 2010's hours recovered and the retired sensors located (the 16 October run had 6,173,993 rows and no November 2022 to October 2024).

| | |
|---|---|
| Rows | 7,409,868 (of 7,410,238 read), 118 sensors, 2009-05-01 to 2026-10-22 (local) |
| Rows by source | archive 4,561,860; Wayback 1,180,314; live 1,667,694 |
| Re-houred: September 2010 | 12,240: **every hour of September 2010 is stamped midnight** in the archive. The rows are in hour order by the archive's `ID`, so each sensor-day's 24 rows are numbered 0–23 in that order (`rehour_midnight_days`). The month's mean hourly profile then matches August 2010's (correlation 1.00). |
| Dropped: unplaceable duplicates | 0 |
| Dropped: non-existent spring-forward hour | 370 |
| Rows from sensors with no location | 0. The 18 sensors missing from today's sensor table (e.g. 7 Birrarung Marr, 13 Flagstaff Station; 867,750 rows) are placed from its October 2022 copy. |
| Suspected outage hours | 9,882 |
| Rows without weather | 16,333, the newest days. The weather archive lags by about a week, and the next run fills them. |

Checks on this run:
- In January, the sun is highest and radiation peaks in the local 13:00 hour (daylight time); in July, the 12:00 hour.
- Temperature peaks at 15:00 local in January and 14:00 in July.
- Daytime counts (08:00–20:00, outage hours left out) average 593 an hour in rain (over 1 mm) against 745 when dry.
