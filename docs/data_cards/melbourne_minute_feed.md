# City of Melbourne Pedestrian Counting System: counts per minute

| | |
|---|---|
| **Role** | Within-hour profiles and sub-hour validation (peaks shorter than an hour, event egress). The hourly table stays the training target. |
| **Resolution** | Despite the name, 42 of the 99 sensors report every minute and the other 57 every 5 minutes (checked 27 October 2026). The profiles use 5-minute blocks, which every sensor supports. |
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
| `sensing_datetime` | Start of the minute, UTC. For sensors that report every 5 minutes, the start of the 5-minute block the row covers |
| `sensing_date`, `sensing_time` | The same minute on Melbourne's clock, as published (the file's day is `sensing_date`) |
| `direction_1`, `direction_2`, `total_of_directions` | Pedestrians in each direction and in total |
| `polled_at` | UTC time of the first poll that saw this row (added by the archive) |

## What was found on 25 October 2026

- The feed holds far more than the "past hour" in its title. At 16:04 Melbourne time on 25 October it held 24,078 rows from 99 sensors, starting at 00:55 that day. The start stayed put while the end moved on.
- **It resets once a day at about 02:00 Melbourne time, not at midnight.** At 01:25 on the 26th it still held all of the 25th and the new day's first hour (42,073 rows). Between 02:00 and 02:20 it emptied (a query at about 02:10 came back empty) and restarted from 00:55 on the 26th. So a day's last minutes and the next day's first 55 minutes are published only until that reset. One poll between midnight and about 02:00 Melbourne time catches both: 11 am in Boston until 1 November, 10 am after, and an hour later again once Melbourne's daylight saving ends in April. The archive's 26 October file starts at 00:00 (981 rows before 00:55); the 25th starts at 00:55 because the archive began after that day's reset.
- **Minutes with no one counted are absent, not zero.** No row that day had a total of 0 (by 27 October there were two, both sensor 184 at 00:50). A missing minute is either a zero or no reading; summing a sensor's minutes and comparing with its hourly count tells which.
- **Two different rows for one sensor and minute** happen (2 of 24,078 rows that day; the portal's description mentions sensors 67-69). The archive keeps both as published and merges only rows identical in every column. Three days later most of these turned out to be revisions, not duplicates (below).
- `direction_1 + direction_2 = total_of_directions` on every row.

## What was found on 27 October 2026

By then the archive held 158,753 rows over three and a half days, and the hourly table overlapped it for 4,675 sensor-hours, daytime included. The first run's overlap was three night hours, too quiet to show the first two findings.

- **57 sensors report every 5 minutes.** All their rows fall on minutes :00, :05, :10 and so on; the other 42 sensors have rows in every minute. Each 5-minute row covers the 5 minutes that start at its timestamp: summed that way, their hours match the hourly table 99.7% of the time, against 67% if the row is taken to cover the 5 minutes before it. A per-minute reading of these sensors puts each block's people in one minute and none in the four after it.
- **Eight per-minute sensors never publish minutes :00-:04.** In every busy hour, on every day, sensors 14 (`SanBri_T`), 31 (`Lyg161_T`), 42 (`UM1_T`), 50 (`Lyg309_T`), 52 (`Eli263_T`), 56 (`Lon364_T`), 180 (`Mac330_T`) and 187 (`Swa330_T`) have no rows for the first 5 minutes of the hour, so their sums run 7-9% short of the hourly count (5 of 60 minutes is 8.3%). The other 34 per-minute sensors match the hourly table in every busy hour. Nothing in the archive can recover those minutes, so these sensors are left out of the profiles and the hourly-sums check.
- **The portal revises some minutes after publishing them.** 40 minutes in three and a half days got a different value in a later poll: 22 went up, 18 down, between 23 minutes and 6.4 hours after they first appeared. The archive keeps both values and the profiles use the latest. Two rows for one minute within a single poll, the true duplicates, happened 4 times.
- **Minutes appear 20-45 minutes late.** The newest minute in each poll was 21 to 43 minutes old, so polling every 15 minutes misses nothing.
- **Event crowds show up.** Sensor 161 (`BirArt1109_T`, Birrarung Marr, between Rod Laver Arena and the city) recorded each night's concert crowd leaving: about 580 people between 23:05 and 23:20 on 25 October (Hans Zimmer), 950 between 22:45 and 23:10 on the 26th and 1,390 between 22:55 and 23:20 on the 27th (Hilary Duff), with 452 in the single block at 23:00, nearly all walking one way.

## What the first ten days show (25 October to 3 November 2026)

Checked on 4 November 2026, with ten full days archived plus the morning of the 4th. Each event is compared with the median of the same weekday and hour from 1 August to 24 October 2026 in the hourly table, over the sensors present in both. The eight sensors with missing minutes and sensor 17 (below) are left out. The minute archive gives the shape inside each hour. Event dates and start times come from the venues' listings in `configs/events.yaml` (on the extension PR, #38), and the weather from Open-Meteo's hourly data for central Melbourne. Each event happened once or twice, so these are case studies, not averages.

- **The archive is complete.** No day or 5-minute stretch is missing, the minutes add up to the hourly table for 99.9% of 18,842 sensor-hours, and the 57/42 split between 5-minute and per-minute sensors and the eight sensors with missing minutes are unchanged. The portal revised 162 minutes after publishing them, and 20 minutes had two rows in one poll.
- **Concerts at Melbourne Park show up at every show.** Five sensors between the arenas and the city (161, 136, 123, 124 and 75, 470-800 m away) carry each crowd leaving, in a narrow window and nearly all one way: at sensor 161 on 27 October, 1,417 people walked toward the city between 22:50 and 23:30 and 4 walked away. A quiet night brings about 200 people past these five sensors between 21:00 and 00:30.

  | Night | Event | Crowd leaving | People | Busiest 5 minutes |
  |---|---|---|---|---|
  | Sun 25 Oct | Hans Zimmer, Rod Laver Arena | 23:05-23:25 | 616 | 279 |
  | Mon 26 Oct | Hilary Duff, Rod Laver Arena | 22:45-23:15 | 1,208 | 374 |
  | Tue 27 Oct | Hilary Duff, Rod Laver Arena | 22:55-23:25 | 1,806 | 576 |
  | Wed 28 Oct | Netball at John Cain Arena, then Lily Allen | 21:05-22:45 | 2,188 | 553 |
  | Thu 29 Oct | Lily Allen, Rod Laver Arena | 22:20-22:45 | 1,275 | 478 |
  | Sun 1 Nov | Richie Jen, Margaret Court Arena | 22:45-22:55 | 289 | 152 |
  | Mon 2 Nov | NBL at John Cain Arena, then 5 Seconds of Summer | 21:35-23:35 | 1,179 | 338 |

  Arrivals show too: on show nights the two hours from 18:00 to 19:59 carried 2.7 to 5.4 times their usual number of people at these sensors, and 1.9 times on the rain-soaked Sunday of Hans Zimmer. Where two events shared a night, their crowds left in two waves.
- **Crowds leaving an event are packed into a quarter of an hour.** In the busiest hour of each crowd leaving, `peak15` at these five sensors was 2.1 to 3.9, against a median of 1.27 (90th percentile 1.89) over their 146 other hours with at least 100 people. An hourly forecast spread evenly over the hour would put the 15-minute peak after a show 2 to 3 times too low, which is why the walkers and the microsimulation should start from measured event profiles.
- **Halloween (Saturday 31 October) was quieter by day and busier at night.** From 09:00 to 16:59 the network carried 60-75% of a usual Saturday, because 11.3 mm of rain fell between 09:00 and 15:00 (rainy Monday 2 November dipped the same way). From 19:00 to 23:59 it carried 1.28 times a usual Saturday, against 1.07-1.12 on the dry weekday evenings of the same week, peaking at 1.4-1.5 times at 22:00-23:00. Sunday 1 November evening was as high (1.27), so the whole Halloween weekend was busy at night, and part of the rise is the season: the baseline is mostly winter Saturdays. The evening rose most at St Kilda Rd by Alexandra Gardens (sensor 29, 1.8 times a usual Saturday evening), Flinders St by ACMI (72, 1.7), Bourke and Spencer St (58, 1.6), Southbank Promenade (35, 1.5) and Harbour Esplanade (77, 1.5), where one of the city's Halloween events was held.
- **Melbourne Cup Day (Tuesday 3 November, a public holiday) looked like a holiday, not an event.** The day carried 77% of a usual Tuesday: 24% of the morning commute (07:00-08:59), about 75% at midday and in the evening peak, and a normal evening after 19:00 (1.06). There is no clear dip when the race starts at 3 pm: the 15:00 block was 5% below that afternoon's average, no lower than other quiet blocks. What is missing is the lift from 16:00 that working days show (10-17% on 27 and 28 October).
- **Heavy rain cut foot traffic by more than half.** On Sunday 25 October, 41.5 mm of rain fell over 17 hours, and the network carried 318,737 people, against about 850,000 on the Sundays before (39%), in both the minute feed and the hourly table. The Hans Zimmer crowd still left Rod Laver Arena at 23:05.
- **The Melbourne Park sensors' Saturday baseline is inflated** by the AFL nights at the MCG in August and September, so Saturdays at those sensors aren't compared here.

## In the pipeline

- **Preprocessing.** `pull_minute_feed` in `stoa_pipeline` copies the day files into `data/raw/melbourne_portal/minute_feed/` for DVC. `minute_profiles` then builds the within-hour profiles on 5-minute blocks. It keeps sensor-hours whose minutes add up to the hourly table's count (within 5%, or 2 people), which also shows that the missing minutes were zeros, and that have at least 30 people, at sensors that publish every block of the hour. It writes `minute_sensors.parquet` (each sensor's resolution, the blocks it never publishes, and whether it is used), `minute_hour_status.parquet` (every sensor-hour in the archive's span and what became of it), `minute_hours.parquet` (`peak15`: four times the busiest clock quarter over the hour, 1 for an even hour, 4 when everyone came in one quarter; `top_block_share`: the busiest 5-minute block's share) and `minute_profiles.parquet` (the mean share of each 5-minute block, by sensor, weekday or weekend, and local hour).
- **Validation.** `validate` warns on missing days, days whose last minute is before 23:00, duplicate rows within a poll, more sensors with never-published blocks than the eight known, and minutes that don't add up to the hourly counts.
- **Bias.** `minute_coverage` measures which slices the profiles under-cover, and pools quiet hours by volume tier as the mitigation (`docs/bias/bias_slicing.md`, "The per-minute data").
- **Three edges are left out.** The archive's first hour when it began part-way through (00:55 on 25 October 2026), and its newest hour, which may still be filling. The hourly table's newest hour is published unfinished: on 25 October its 03:00 counts were below the minutes' sums at 85% of sensors, until the next pull replaced them.

## Known issues

- Polls can be late or missed (a queue wait on the cluster, a sleeping laptop). That costs nothing as long as one comes between midnight and the reset at about 02:00 Melbourne time; without one, the archive keeps only what earlier polls saw of the day before and none of the new day's first hour.
- The portal's description is from 2020 and still mentions an issue with sensors 67-69. Those three sensors matched the hourly table in every hour checked on 27 October.
- Sensor 17 (Collins Place South, about 14,000 people a day) stopped reporting at 11:58 on 29 October 2026, in the minute feed and the hourly table alike, so the sensor is down rather than the poll. It was still silent on 4 November; `validate`'s completeness check reports it.
- Eight sensors never publish minutes :00-:04 (above). The check is over the whole archive, so if the portal starts publishing those minutes, the sensors stay excluded until the rule is changed to look at recent days only.
