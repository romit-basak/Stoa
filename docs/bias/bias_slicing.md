# Bias analysis by data slicing

The data pipeline guidelines ask us to slice the data, check whether some groups are served worse than others, mitigate what we find, and write it down. This page covers the slices and step 1, the representation report. Steps 2 to 4 come later (see the end).

## What "bias" means here

Stoa holds no personal data (CLAUDE.md, core rule 6): the counts are anonymous totals per sensor and hour. There are no age, gender or other demographic fields to slice on, by design. The groups that matter for a pedestrian model are places, times and conditions. A model trained mostly on busy CBD streets in fine weather can look accurate overall and still be poor on a quiet street in Kensington, at night or in the rain. Planners would then get worse forecasts for exactly the places that have the least data.

## The slices

Arshmehar proposed the slice list (her table, and her notebook `notebooks/04_bias_slicing.ipynb`), which also gave the held-out sensor lists, the sensor start period, the count-distribution columns, coverage by year and the cross-tables. The calendar, event, density and sensor-quality slices and the reference comparison were added on top. The **Proposed by** column says which is which. Settings are in `configs/bias.yaml`; the code is `src/bias/slices.py`.

| Slice | Groups | Why it could hide bias | Built from | Proposed by |
|---|---|---|---|---|
| `precinct` | CLUE areas, with the three CBD areas merged | Sensors cluster in the CBD | Sensor location in the City's CLUE areas | Arshmehar (CLUE grouping added) |
| `holdout_precinct` | UniMelb edge (42–44), Carlton (31, 37, 46, 50, 54), North Melbourne (70, 76, 85, 86, 87, 180), the rest | These are the within-Melbourne transfer test | The sensor lists in `docs/PROJECT_CONTEXT.md` 7.1, copied to `configs/bias.yaml`. They don't follow CLUE boundaries: Macaulay Rd (76, 85) is in the Kensington area. | Arshmehar |
| `volume_tier` | Quiet, medium, busy thirds of sensors | Errors on busy sensors dominate absolute metrics | Each sensor's mean hourly count | Arshmehar |
| `amenity_tier` | Low, medium, high thirds | A density proxy (CLAUDE.md asks for error by density) | OSM amenities within 400 m of the matched segment | Added |
| `slope_band` | <1%, 1–3%, 3–5%, ≥5% | Steep paths are rare | `sensor_steepness.parquet` | Arshmehar |
| `slope_source` | Footpath survey, DEM | The 2 DEM sensors use a different method | `place_source` | Arshmehar |
| `sensor_start` | 2009–2013, 2014–2018, 2019 on | Early years rest on a handful of streets | Year of each sensor's first count | Arshmehar |
| `time_of_day` | Night 0–6, morning peak 6–10, midday 10–16, evening peak 16–20, late evening 20–24 | Most pedestrians walk in the day, so night errors are easy to miss | `hour_local` | Arshmehar |
| `day_type` | Weekday or weekend | Different walking patterns | `is_weekend` | Arshmehar |
| `public_holiday`, `school_term`, `uni_teaching` | Each calendar flag | Different walking patterns | The calendar columns (`unknown` where a calendar doesn't cover the year) | Added |
| `rain` | Dry (<0.1 mm), light, rain (≥1 mm) | Rainy hours are rare | `wx_precipitation` | Arshmehar |
| `temperature` | Cold (<10 °C), mild, warm (25–30 °C), hot (≥30 °C) | Hot hours are rare | `wx_temperature_2m` | Arshmehar |
| `era` | Pre-COVID, COVID (16 Mar 2020 to 21 Oct 2021), recovery, post Metro Tunnel (from 30 Nov 2025) | Crowds changed sharply | `date_local` | Arshmehar |
| `afl_day` | Marvel, MCG, both, none | Event days are reported separately (CLAUDE.md, "Evaluation") | AFL Tables attendance | Added |
| `source` | Archive, live (Wayback once it's in the pipeline) | Different collection | `source` | Arshmehar |
| `missing_hours` | <5%, 5–20%, ≥20% of hours absent | Unreliable sensors can skew training and evaluation | Hours absent within each source's span for the sensor | Arshmehar |
| `match_review` | Matched automatically, needs review | A sensor on the wrong segment gets the wrong place features | `needs_review` from the sensor match | Added |
| `has_location` | Located, no location | The 18 retired sensors have no place features yet | `has_location` | Added |

The cut-offs for rain, temperature and sensor tiers are agreed: Arshmehar's first notebook used wet above 0.2 mm, bands at 10/20/30 °C and the median count, and her notebooks now use the ones above.

Slices are for checking, never features. Precinct, sensor ID, season and month must not go into a model (CLAUDE.md core rule 9); they only group rows so results can be compared.

Rows flagged `suspect_outage_zero` are dropped before the shares are computed, as in training. Each group still reports the share of its rows that were flagged.

## Step 1: the representation report

```bash
.venv/bin/python -m src.bias.representation    # ~35 s
```

It runs as `bias_slices` in the DAG, next to `validate`. It needs no model, and writes to `data/processed/bias/melbourne/`:

| File | One row per |
|---|---|
| `slice_representation.csv` | Slice and group |
| `slice_crosstabs.csv` | Pair of groups, for the pairs in `crosstabs` (time of day × rain, precinct × era) |
| `coverage_by_year.csv` | Year: rows, sensors, pedestrians |
| `sensor_slices.csv` | Sensor: its slices and the values behind them (mean count, first year, missing share, slope) |
| `slice_representation_meta.json` | The run: row counts, period, thin groups |

Each group gets its number of sensors and rows, its share of rows and of pedestrians, its mean, median and 90th-percentile count, its share of zero counts and its outage share. Where a reference exists, it also gets a **ratio**: the group's share of the data divided by its share of the reference.
- Time, calendar, weather, era and AFL slices are compared with every hour of the study period. A ratio below 1 means outages, or the sensor network's growth, left that kind of hour short.
- Precincts are compared with the length of OSM walkable network in each CLUE area, using the share of sensors. A ratio below 1 means the precinct has fewer sensors than its share of the network.
- Slope bands are compared with the length of surveyed footpath in each band.

A group is flagged **thin** when it has fewer than 5 sensors or a ratio below 0.5.

The figures below come from `notebooks/04_bias_slicing.ipynb`, which reads these files and `configs/bias.yaml` and writes PNGs to `docs/bias/figures/` (it defines no slices of its own). `notebooks/05_bias_stage1.ipynb` runs the same representation check, with the same cut-offs, on the notebooks' own dataset, which already includes the Wayback fill for November 2022 to October 2024.

### Findings (run of 22 October 2026)

6.2M sensor-hours from 118 sensors, May 2009 to October 2026.

**Place is where the data is most uneven.**

| Precinct | Sensors | Share of sensors | Share of walkable network | Ratio |
|---|---|---|---|---|
| CBD | 60 | 60% | 12% | 4.95 |
| Docklands | 11 | 11% | 6% | 1.74 |
| Southbank | 5 | 5% | 5% | 0.92 |
| West Melbourne (Residential) | 4 | 4% | 4% | 0.94 |
| Melbourne (Remainder) | 5 | 5% | 7% | 0.78 |
| North Melbourne | 4 | 4% | 10% | 0.40 |
| Carlton | 6 | 6% | 15% | 0.39 |
| Parkville | 3 | 3% | 10% | 0.30 |
| Kensington | 2 | 2% | 11% | 0.18 |

The CBD has five times its share of sensors; Carlton, North Melbourne, Parkville and Kensington have well under half of theirs. Three of those are held-out transfer precincts, with 3 to 6 sensors each, so their errors will carry wide uncertainty.

![Share of sensors against share of the walkable network, by precinct](figures/precinct_sensors_vs_network.png)

**Time, calendar and weather are represented in proportion.** Every time-of-day, weekday, holiday, school-term, rain, temperature and AFL group has a ratio between 0.93 and 1.06. Outages are not concentrated in particular hours or weather. Some groups are still small in absolute terms: rain of 1 mm or more is 2.5% of rows, and 30 °C or hotter is 2.0%.

![Ratio of each group's share of rows to its share of hours](figures/ratio_by_group.png)

Rain lowers counts at every time of day (median 471 dry against 326 in rain at midday), so the weather slices behave differently even though they're represented in proportion.

![Median count by time of day and rain](figures/heatmap_time_of_day_x_rain.png)

**Night has rows but few pedestrians.** The night slice (0–6) is 24% of rows and 2.8% of pedestrians. An error metric in absolute numbers would barely register night errors, so step 2 uses relative errors.

![Share of rows and of pedestrians by time of day](figures/time_of_day_rows_vs_pedestrians.png)

**Later eras are over-represented** because the sensor network grew. Pre-COVID is 62% of the hours but 49% of the rows (ratio 0.79); the post Metro Tunnel era is 5% of the hours and 12% of the rows (ratio 2.3). The recovery era's reference also counts the November 2022 to October 2024 gap, which has no rows until the Wayback fill is in the pipeline.

![Sensors and rows per year](figures/coverage_by_year.png)

Early years rest on 18 sensors (2009–2012, about 2% of rows each), against 102 in 2025. Kensington, North Melbourne and West Melbourne (Residential) have no rows before COVID at all, so a model can't learn their pre-2020 pattern. COVID cut counts in every precinct, and of the six precincts with pre-COVID data only Docklands is back above its pre-COVID median.

![Median count by precinct and era](figures/heatmap_precinct_x_era.png)

**Other points:**
- The busiest third of sensors carries 72% of pedestrians.
- 13% of rows (17.5% of pedestrians) come from the 18 retired sensors with no location, so they're missing from every place slice.
- The 3–5% slope band has 14 sensors against 21% of surveyed footpath length (ratio 0.66). Only 2 sensors use the DEM slope, too few to judge on their own.

  ![Share of sensors and of footpath length by slope band](figures/slope_sensors_vs_footpath.png)
- 10 sensors are missing 20% or more of their hours. The least complete are quiet sensors: 73 (67% of hours missing), 124 (55%), 140 (42%) and 138 (40%), so quiet streets are short of data twice over.
- University teaching is unknown for 16% of rows: there's no calendar before 2015.
- 45 of the 100 located sensors have a sensor match flagged for review.

## Types of bias found

There are no demographic fields: Stoa holds no personal data (CLAUDE.md, core rule 6), so bias here means places, times and conditions. Sensor counts are given for small groups; conclusions are not drawn from 2–3 sensors (the DEM sensors, UniMelb edge, Kensington).

| Type | What the data shows | Slices |
|---|---|---|
| **Sampling bias by place** (where sensors were put) | The CBD has 60 of 100 located sensors against 12% of the walkable network (ratio 4.95). Carlton (6 sensors), North Melbourne (4), Parkville (3) and Kensington (2) have under half their share. | `precinct`, `holdout_precinct` |
| **Volume skew** | The busiest third of sensors (39) carries 72% of pedestrians; the quietest third (40) 6%. The least complete sensors are quiet ones. | `volume_tier`, `missing_hours` |
| **Coverage over time** | 18 sensors in 2009–2012, 102 in 2025. Pre-COVID hours are under-represented (ratio 0.79) and post Metro Tunnel hours over-represented (2.32). November 2022 to October 2024 has no rows in the pipeline yet. Kensington, North Melbourne and West Melbourne (Residential) have no pre-COVID rows. | `era`, `sensor_start`, coverage by year |
| **Era shifts** | Median count 250 pre-COVID, 72 during COVID, 153 in recovery; only Docklands is back above its pre-COVID median. | `era`, precinct × era |
| **Where the pedestrians are in time** | Night is 24% of rows but 2.8% of pedestrians. | `time_of_day` |
| **Rare conditions** | Rain of 1 mm or more is 2.5% of rows, 25–30 °C 4.5%, 30 °C or hotter 2.0%. Only 2 sensors use DEM slope. | `rain`, `temperature`, `slope_source` |
| **Data quality** | 18 retired sensors (13% of rows, 17.5% of pedestrians) have no location; 45 of the 100 located sensors have a match flagged for review; 10 sensors miss 20% or more of their hours. | `has_location`, `match_review`, `missing_hours` |

**Not found:** time-of-day, weekday, holiday, school-term, university-term, rain, temperature and AFL groups are each represented in proportion to the hours they cover (ratio 0.93 to 1.06).

## How it was detected

1. **Slices.** The hourly table is split by each slice in the table above (`src/bias/slices.py`, settings in `configs/bias.yaml`). Outage-flagged rows are dropped first, as in training.
2. **Representation.** For each group: rows, sensors, share of rows and of pedestrians, mean, median and 90th-percentile count, zero and outage shares (`src/bias/representation.py`).
3. **Reference ratio.** Each group's share is compared with what it would be if the data were spread fairly: the hours in the study period for time and condition slices, the walkable network for precincts, surveyed footpath length for slope bands. A group is **thin** with fewer than 5 sensors or a ratio below 0.5.
4. **Cross-tables and coverage.** Time of day × rain and precinct × era show problems that appear only where two slices meet; coverage by year shows how the sensor network grew.
5. **Second dataset.** `notebooks/05_bias_stage1.ipynb` repeats the check, with the same cut-offs, on the notebook dataset, which includes the Wayback fill for November 2022 to October 2024. The same groups stand out there.

Step 1 measures the **data**. Whether a model is actually worse on these groups is step 2: until then, each finding is a candidate.

## Mitigation plan (step 3)

Each row applies only if step 2 shows the model's error on that slice is clearly worse than overall. What's done, and what it costs, is recorded here.

| If the model is worse on... | Planned mitigation |
|---|---|
| Thin precincts and held-out precincts | Weight training rows so each precinct (or each sensor) counts equally in the loss, so the CBD doesn't dominate. Report the held-out precincts separately, with their sensor counts and confidence intervals from resampling whole sensors. |
| Quiet sensors | Use relative error (WAPE, or error on log counts) so busy sensors don't dominate the metric; weight sensors equally. |
| Night | Relative error, and night reported separately. |
| Early years, short sensor histories | Weight by sensor-year, or compare with a model trained on a recent window only, and say which is used. |
| COVID and post Metro Tunnel | Keep the `covid_period` and `post_metro_tunnel` flags; report post Metro Tunnel results separately (CLAUDE.md, "Structural break"). |
| Rain, heat, DEM sensors | Report these groups separately; weight their rows up if their error is much worse. Don't present an effect learned from 2 sensors as general. |
| Missing years | Data fix: bring the Wayback fill into the pipeline (adds November 2022 to October 2024). |
| Sensors with no location | Data fix: locations for the 18 retired sensors from the October 2022 copy of the City's sensor table. |
| Incomplete sensors and flagged matches | Report sensors missing 20% or more of their hours separately; hand-check the 45 flagged matches. |
| Outage zeros | Already dropped from training and from these shares. |

## Trade-offs

- **Reweighting** gives small precincts and quiet sensors more say, which usually costs some accuracy on the busy CBD sensors, and can raise the overall error. Overall and per-slice errors are reported before and after.
- **Relative error metrics** make night and quiet-hour errors visible, but make busy-hour errors look smaller in people terms. Both kinds of metric are reported.
- **Dropping or setting aside data** (outage rows, incomplete sensors) removes rows from groups that are already thin, mostly quiet sensors.
- **Small groups** give wide confidence intervals. Tuning a model to 2–3 sensors would overfit them, so those groups are reported, not optimised for.
- **The Wayback fill** adds about two years from an archived copy of the City's own export, which matched the archive on about 98.5% of overlapping hours. The `source` slice keeps it checkable.

## Next steps

1. **Per-slice errors (step 2).** Train a simple baseline with whole sensors held out (gradient boosting, folds grouped by sensor) and report each slice's error with Fairlearn's `MetricFrame` (MIT). Use relative error (WAPE, or error on log counts) so busy slices don't dominate. Give confidence intervals by resampling whole sensors, because hours from one sensor aren't independent.
2. **Mitigation (step 3).** Where a slice's error is much worse, try reweighting its rows (for example by precinct, so the CBD doesn't dominate the loss) and record what it costs in overall accuracy.
3. **Rerun on the real model (step 4)** once the flow model exists.
4. **Data fixes that change the report:** bringing the Wayback fill and the retired sensors' locations into the pipeline (Arshmehar's and Shubham's work) adds the 2023–24 rows and puts the 18 sensors into the place slices.
