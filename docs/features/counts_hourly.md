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

## Time handling

People follow the local clock, so **hour, weekday and every calendar column use Melbourne local time, daylight saving included.** Physical quantities use the real instant: weather is joined on UTC, and sun elevation is computed from UTC.

The sources don't agree on clocks, so this matters:

| Source | Clock | How we know |
|---|---|---|
| Counts, archive (2009–2022) | Local, with daylight saving | The weekday morning peak is at 8 a.m. in both January and July |
| Counts, live (2024–now) | Local, with daylight saving | The October changeover day has 23 hours |
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
| `source` | `archive` (2009-05 to 2022-12) or `live` (2024-10 onward) |
| `pedestriancount` | Count for the hour (the target) |
| `direction_1`, `direction_2` | Directional counts: live window only, NaN in the archive. They add up to `pedestriancount`, so **never use them as features**: they would leak the target. |
| `suspect_outage_zero` | In a run of ≥ 3 consecutive zero hours that touches 08:00–20:00. Likely a sensor fault, not an empty street. Drop these rows when training. |
| `is_public_holiday` | Victorian public holiday (`holidays` package) |
| `school_term` | Within a Victorian school term (student days; weekends included). Missing outside 2009–2026. |
| `uni_teaching`, `uni_exams` | University of Melbourne teaching weeks (breaks excluded) and exam periods. **Missing before 2015** (no calendar), not False. |
| `sun_elevation_deg` | Sun above the horizon, in degrees, at the middle of the hour (NOAA formula, city centre) |
| `covid_period` | 16 March 2020 to 21 October 2021: flagged, not dropped |
| `post_metro_tunnel` | On or after 30 November 2025 (structural break) |
| `wx_*` | Open-Meteo ERA5 for the city centre: `temperature_2m`, `apparent_temperature`, `relative_humidity_2m`, `cloud_cover`, `wind_speed_10m`, `is_day`, `precipitation`, `rain`, `shortwave_radiation` |
| `sensor_description`, `location_type`, `segment_id`, `match_method`, `needs_review` | From the sensor match (`sensor_segment_map.parquet`) |
| `has_location` | The sensor is in the current sensor-locations file |
| `slope_pct`, `slope_pct_local_mean`, `elevation_m`, `place_source` | Footpath slope (%), the length-weighted mean slope within 25 m, and elevation (m AHD), from `sensor_steepness.parquet` (below). Missing for the 18 sensors with no location. |

There's no month or day-of-year column, by design (core rule 9): use `sun_elevation_deg`, temperature and the calendar flags instead.

## Footpath slope per sensor

`src/features/sensor_steepness.py` (task `sensor_steepness`) writes `sensor_steepness.parquet`, one row per outdoor sensor. It's based on Shubham Hadawle's Colab join notebook. Each sensor is matched to the nearest section of the City of Melbourne footpath-steepness survey within 25 m:
- `slope_pct` is that section's `gradepc` (metres of rise per 100 m along the footpath).
- `elevation_m` is the mean of its `rlmin` and `rlmax`.
- `slope_pct_local_mean` is the length-weighted mean `gradepc` of all sections within 25 m, so one odd section doesn't decide the value.

Sections with no grade (4,455) or shorter than 5 m (4,854; their grades reach 580%) are skipped. Skipping the ungraded ones matters: without it, sensor 143 was matched to an empty section and got no slope.

Sensors 25 and 130 have no section within 25 m. Their elevation and slope come from the GA 5 m LiDAR DEM, with slope taken from samples 10 m to each side. That's the steepest terrain gradient, not the grade along a footpath, so `place_source` (`footpath_steepness` or `ga_dem_5m`) records which method each sensor used.

Sensor 139 (Sandridge Bridge signal box, by a ramp) has 20.4%; every other sensor is under 10%. The footpath survey exists only for the City of Melbourne, so these columns belong to the Melbourne-rich model, not the portable one (CLAUDE.md, "Generalization").

## Run of 16 October 2026

| | |
|---|---|
| Rows | 6,173,993 (of 6,186,603 read), 118 sensors, 2009-05-01 to 2026-10-03 |
| Dropped: unplaceable duplicates | 12,240: **every hour of September 2010 is stamped midnight** in the archive, so those days can't be put in order |
| Dropped: non-existent spring-forward hour | 370 |
| Rows from sensors with no location | 817,199 (18 sensors, e.g. 7 Birrarung Marr, 13 Flagstaff Station). They have no `segment_id`. |
| Suspected outage hours | 9,666 |
| Rows without weather | 13,928, all from 2026-09-27 to 2026-10-03. The weather archive lags by about a week, and the next run fills them. |

Checks on this run:
- In January, the sun is highest and radiation peaks in the local 13:00 hour (daylight time); in July, the 12:00 hour.
- Temperature peaks at 15:00 local in January and 14:00 in July.
- Daytime counts average 623 an hour in rain (over 1 mm) against 800 when dry.
