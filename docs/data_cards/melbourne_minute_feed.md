# City of Melbourne Pedestrian Counting System: counts per minute

| | |
|---|---|
| **Role** | Within-hour profiles and sub-hour validation (peaks shorter than an hour, event egress). The hourly table stays the training target. |
| **Provider** | City of Melbourne Open Data, `pedestrian-counting-system-past-hour-counts-per-minute` |
| **License** | CC BY (the dataset's licence field). Credit "Source: City of Melbourne Open Data". |
| **Access** | `python -m src.ingest.minute_feed --sync` copies the archive into `data/raw/melbourne_portal/minute_feed/` (needs bucket credentials, like `dvc pull`). |
| **Archive** | The portal keeps only the current Melbourne day, so the `stoa_minute_feed` DAG (`dags/stoa_minute_feed.py`) polls it every 15 minutes and merges each poll into one parquet file per Melbourne day in `gs://stoa-dvc-data/minute_feed/`. The same poll runs every 15 minutes on Northeastern's Explorer cluster through Slurm's `scrontab` (`ops/hpc/`), so the archive doesn't depend on anyone's laptop being awake. Each write goes through only if the file hasn't changed since it was read, so several machines can poll at once. Archived from 2026-10-25; nothing earlier exists anywhere. |
| **Update cadence** | The portal says every 15 minutes. |
| **Poll cost** | The portal takes about 8 s to export the whole day so far but about 2.5 s for the last 3 hours, so each hour's first poll fetches everything (catching any minute published late) and the other three fetch the last 3 hours. Timed on 25 October 2026: a full poll 7–9 s end to end, a recent one 4.5 s; on the cluster, a job starts about a minute after its slot and runs about 18 s including loading Python. |

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

- The feed holds far more than the "past hour" in its title. At 16:04 Melbourne time on 25 October it held 24,078 rows from 99 sensors, starting at 00:55 that day. The start stayed put while the end moved on.
- **It resets once a day at about 02:00 Melbourne time, not at midnight.** At 01:25 on the 26th it still held all of the 25th and the new day's first hour (42,073 rows). Between 02:00 and 02:20 it emptied (a query at about 02:10 came back empty) and restarted from 00:55 on the 26th. So a day's last minutes and the next day's first 55 minutes are published only until that reset. One poll between midnight and about 02:00 Melbourne time catches both: 11 am in Boston until 1 November, 10 am after, and an hour later again once Melbourne's daylight saving ends in April. The archive's 26 October file starts at 00:00 (981 rows before 00:55); the 25th starts at 00:55 because the archive began after that day's reset.
- **Minutes with no one counted are absent, not zero.** No row has a total of 0. A missing minute is either a zero or no reading; summing a sensor's minutes and comparing with its hourly count tells which.
- **Two different rows for one sensor and minute** happen (2 of 24,078 rows that day; the portal's description mentions sensors 67-69). The archive keeps both as published and merges only rows identical in every column.
- `direction_1 + direction_2 = total_of_directions` on every row.

## Known issues

- Polls can be late or missed (a queue wait on the cluster, a sleeping laptop). That costs nothing as long as one comes between midnight and the reset at about 02:00 Melbourne time; without one, the archive keeps only what earlier polls saw of the day before and none of the new day's first hour.
- The portal's description is from 2020 and still mentions an issue with sensors 67-69.
