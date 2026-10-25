# City of Melbourne Pedestrian Counting System: counts per minute

| | |
|---|---|
| **Role** | Within-hour profiles and sub-hour validation (peaks shorter than an hour, event egress). The hourly table stays the training target. |
| **Provider** | City of Melbourne Open Data, `pedestrian-counting-system-past-hour-counts-per-minute` |
| **License** | CC BY (the dataset's licence field). Credit "Source: City of Melbourne Open Data". |
| **Access** | `python -m src.ingest.minute_feed --sync` copies the archive into `data/raw/melbourne_portal/minute_feed/`. |
| **Archive** | The portal keeps only the latest stretch of minutes, so `.github/workflows/minute_feed.yml` polls it every 15 minutes on GitHub's servers and keeps one parquet file per Melbourne day as an asset of this repo's GitHub Release `minute-feed-<year>`. Archived from 2026-10-25; nothing earlier exists anywhere. |
| **Update cadence** | The portal says every 15 minutes. |

## Schema

One file per local day, `minute_feed_YYYY-MM-DD.parquet`:

| Column | Meaning |
|---|---|
| `location_id` | Sensor; joins the sensor tables and the hourly counts |
| `sensing_datetime` | Start of the minute, UTC |
| `sensing_date`, `sensing_time` | The same minute on Melbourne's clock, as published (the file's day is `sensing_date`) |
| `direction_1`, `direction_2`, `total_of_directions` | Pedestrians in each direction and in total |
| `polled_at` | UTC time of the first poll that saw this row (added by the archive) |

## What was found on 25 October 2026

- One poll held 24,078 rows from 99 sensors, from 00:55 to 16:04 Melbourne time: much more than the "past hour" in the title. Whether the window rolls or restarts each day is being checked from the first days of polls.
- **Minutes with no one counted are absent, not zero.** No row has a total of 0. A missing minute is either a zero or no reading; summing a sensor's minutes and comparing with its hourly count tells which.
- **Two different rows for one sensor and minute** happen (2 of 24,078 rows that day; the portal's description mentions sensors 67-69). The archive keeps both as published and merges only rows identical in every column.
- `direction_1 + direction_2 = total_of_directions` on every row.

## Known issues

- GitHub can start a scheduled run late or skip one under load. The feed holds more than an hour, so the next poll covers it; the daily pipeline will warn about gaps longer than that.
- GitHub turns off scheduled workflows in a public repo after 60 days with no repository activity. Re-enable it from the Actions tab if the archive stops growing.
- The portal's description is from 2020 and still mentions an issue with sensors 67-69.
