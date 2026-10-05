# City of Melbourne Pedestrian Counting System (hourly counts)

| | |
|---|---|
| **Role** | **Target variable.** Training and validation for the flow model; all accuracy claims rest here. |
| **Provider** | City of Melbourne Open Data (`pedestrian-counting-system-monthly-counts-per-hour`, `pedestrian-counting-system-sensor-locations`) |
| **License** | Sensor locations: CC BY 4.0. Counts: CC BY 4.0 under the portal-wide terms ("Our open data is provided under a creative commons licence … free to share and adapt our data for any purpose, provided you give the appropriate credit", About our data page; confirmed 2026-10-04). The dataset's own licence field is empty; every other CoM dataset lists CC BY 4.0. Credit "Source: City of Melbourne Open Data". |
| **Access** | `python -m src.ingest.melbourne_portal --only pedestrian-counting-system-monthly-counts-per-hour pedestrian-counting-system-sensor-locations` |
| **Update cadence** | Daily (rolling window); the archive is static |

## Files (`data/raw/melbourne_portal/`)

| File | Rows | Coverage |
|---|---|---|
| `pedestrian-counting-system-monthly-counts-per-hour.parquet` | 1,624,373 | 2024-10-04 → 2026-10-03, 103 sensors, ~90% of sensor-hours present |
| `counts_per_hour_2009-05_to_2022-12-14.csv.zip` (423 MB unzipped) | 4,562,230 | **2009-05-01 → 2022-10-31** (despite the filename), 82 sensors |
| `pedestrian-counting-system-sensor-locations.geojson` | 134 | 100 outdoor, 34 indoor; all status `A` |

`data/raw/melbourne_pedestrian/` holds the original hand-downloaded CSVs. That counts CSV is a **~9% sparse subset** (148,500 rows) and is superseded by the parquet above. Do not train on it.

## Schemas

**Current (API) export:** `id`, `location_id`, `sensing_date` (local date), `hourday` (0–23, local), `direction_1`, `direction_2`, `pedestriancount` (= dir1 + dir2 on every row), `sensor_name`, `location` (lat/lon).

**Archive:** `ID`, `Date_Time` (text, local), `Year`, `Month` (name), `Mdate`, `Day`, `Time` (0–23), `Sensor_ID`, `Sensor_Name`, `Hourly_Counts`. There are no directional counts.

**Direction:** labels are per sensor in the locations file (`direction_1`/`direction_2`): 53 sensors North/South, 47 East/West, indoor sensors none. Directional counts exist on every live-window row but **not at all in the archive**, so directional data starts 2024-10-04.

**Active periods:** the locations file lists only active sensors (all status `A`; installation date present for 98%). Retired sensors' active periods must be inferred from their first and last count dates.

`Sensor_ID` in the archive equals `location_id` in the current export (spot-checked by name, e.g. 2 = Bourke Street Mall (South), 61 = RMIT Building 14).

## Coverage

| Period | Sensors | Notes |
|---|---|---|
| 2009–2012 | 18 | Core CBD only |
| 2013–2016 | 32–43 | |
| 2017–2022 | 52–73 | |
| **2022-11 → 2024-10** | — | **23-month gap: in neither the archive nor the API window** |
| 2024-10 → now | 103 | 67 of these also appear in the archive; 15 archive sensors are retired |

- 57 sensors have ≥5 years of history in the archive.
- In the current window, 48,791 sensor-days are complete (24 h). The rest are partial outages.
- Spatial extent: lon 144.929–144.986, lat -37.826 to -37.789 (CBD, Southbank, Docklands, Carlton, North Melbourne). This is **not** Greater Melbourne.

## Distribution (current window)

- Hourly mean ≈ 394, median 160, max 6,632. Right-skewed, so model on log1p or use a Poisson/Tweedie loss.
- Daily profile: low at 04:00 (~21/h), lunch peak at 12–13 (~745), evening peak at 17:00 (~800).
- Busiest: Swanston St (Swa123_T, Swa31), Southbank, QV North, Elizabeth/Flinders.
- Zeros: ~0.007% of rows (true zeros are possible; see the portal note).

## Known issues and handling

0. **The live table is a rolling ~2-year window.** On 2026-10-04 the portal started at 2024-10-05, a day later than our 2026-10-03 pull. Re-pulls are merged into the held rows (`rolling_window` in `configs/data.yaml`), never overwritten. A dated copy of the first pull is in `data/raw/melbourne_portal/snapshots/`. Once DVC is set up, every pull is a version.

1. **The 23-month gap** splits training into two eras. Treat this as a natural temporal holdout. Ask the CoM Open Data team (`opendata@melbourne.vic.gov.au`) whether they can supply Dec 2022 – Oct 2024. Nothing public covers it.
2. **Archive duplicates:** 11,730 duplicate (sensor, date, hour) keys, probably from the DST fall-back hour. Aggregate (sum) or drop them with a logged rule.
3. **Retired sensors:** IDs 28, 65, 78 have counts but no metadata row. Recover their coordinates from the archive or older location snapshots, or drop them.
4. **Sensor relocations/replacements:** the location `note` field records device swaps (e.g. 11 replaced 2019-03-22). Check for level shifts at replacement dates.
5. **Outages** appear as missing rows, not zeros. Never impute missing hours as zero.
6. **COVID (2020-03 → 2021-11)** lockdowns are a regime shift. Flag them as a feature or exclude them from the main training set, and report results with and without.
   - 24% of archive rows fall in 2020–21. Only 43 sensors have 5+ non-COVID years. North Melbourne's archive sensors (70, 76, 85–87) start in 2020–22, so that transfer precinct has almost no normal pre-gap history.
   - Live-window completeness (share of hours present over each sensor's active span): median 99%. Five sensors are under 70%: IDs 1, 123, 124, 138, 140. Down-weight or exclude them.
7. **Split by location (rule 2):** with ~100 outdoor sites, use grouped K-fold over sensors (spatially blocked, so neighbouring sensors on the same street don't straddle folds). Never split by time alone.
8. Indoor sensors (34: libraries, Fitzroy Gardens, 510–516 Elizabeth St) have no counts in this dataset and are out of scope anyway (no venue interiors).

## Structural break: Metro Tunnel (from 2025-11-30)

The Metro Tunnel opened on 2025-11-30, with the full timetable from 2026-02-01. Stations: Arden, Parkville (Grattan St, by the University of Melbourne), State Library and Town Hall (both on the Swanston St corridor), and Anzac. Town Hall's Swanston St / Flinders St entrance opened on 2026-01-29.

- Expect level shifts at Swanston St sensors and at the **University of Melbourne edge sensors (42-44, Grattan St / Swanston St)**, which are also a transfer test area.
- The 2019 CoM pedestrian network has no station entrances. Add them, with opening dates, to the graph and transit features. Otherwise post-opening flows look unexplained.
- Report results before and after the opening separately, or include a station-access feature that changes on those dates.
