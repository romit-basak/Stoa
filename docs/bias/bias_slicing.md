# Bias analysis by data slicing

The data pipeline guidelines ask us to slice the data, check whether some groups are served worse than others, mitigate what we find, and write it down. This page covers the slices, step 1 (how much data each group has), step 2 (where a model's error is worse) and step 3 (what was tried to fix it, and what it cost). Step 4 repeats steps 2 and 3 on the real flow model once it exists.

**Summary.**
- The data is most uneven by place: the CBD has 5 times its share of sensors, and the outer precincts under half of theirs.
- On held-out sensors, a baseline model is clearly worse on seven groups: quiet sensors, low-amenity sensors, steep paths (5% or more), sensors whose path match needs review, night, public holidays and the COVID era. The thin outer precincts are not among them.
- Reweighting the training rows lowered the average error but left those seven groups as they were, so it was rejected. Calibrating each sensor to four weeks of its own counts cut the overall error by a third and fixed the place and volume groups. Night and COVID are still worse; they need changes to the model, not to the weights.

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

## Step 2: where a model is worse

```bash
.venv/bin/python -m src.bias.slice_errors     # ~11 min; the DAG runs it weekly
```

Step 1 found where the data is thin. Step 2 checks whether a model is actually worse there. The real flow model doesn't exist yet, so a baseline stands in for it (`src/bias/slice_errors.py`):
- **Model:** scikit-learn gradient boosting (`HistGradientBoostingRegressor`, Poisson loss) on time, calendar, weather, slope and OSM segment features. Following core rule 9, it has no sensor or segment IDs, no coordinates and no month or day of year.
- **Held-out sensors:** 5 folds grouped by sensor (core rule 2), so every one of the 100 located sensors gets predictions from a model that never saw it. Each fold trains on 600,000 sampled rows.
- **Error:** WAPE, the sum of absolute errors divided by the sum of counts, computed per slice group with Fairlearn's `MetricFrame` (MIT). A relative error keeps busy sensors from drowning out quiet ones, and night (2.8% of pedestrians) from disappearing.
- **Uncertainty:** a 95% interval for each group, from resampling its sensors 300 times. Hours from one sensor aren't independent, so resampling rows would overstate the certainty.
- **Worse:** a group is marked worse when its whole interval lies above the overall WAPE.
- **Scored hours:** each sensor's first 28 days are left out of scoring, because step 3's calibration uses them. That leaves 5.3M sensor-hours.

It runs as `bias_errors` in the DAG, after `bias_slices`. It reruns when its last run is a week old or its settings changed, and otherwise finishes at once. Outputs, in `data/processed/bias/melbourne/`:

| File | One row per |
|---|---|
| `slice_errors.csv` | Mitigation, slice and group: WAPE with its interval, bias, MAE, sensors, rows, `worse`, and the slice's disparity ratio (lowest group WAPE / highest) |
| `mitigation_summary.csv` | Mitigation: overall WAPE, bias and MAE, and which groups are worse |
| `slice_errors_meta.json` | The run: features, folds, model settings |

### Findings (run of 22 October 2026)

Overall WAPE is **0.58**, and the baseline under-predicts total volume by 31%. Without any local counts it gets the shape of the day right but not how busy a place is, which is what per-site calibration is for (CLAUDE.md, "Generalization").

**Seven groups are clearly worse:**

| Group | Sensors | WAPE | 95% interval | Bias |
|---|---|---|---|---|
| Quiet sensors (lowest third by mean count) | 35 | 0.78 | 0.63–1.03 | +36% |
| Steep paths (5% or more) | 13 | 0.76 | 0.60–0.94 | −33% |
| Low-amenity sensors | 34 | 0.71 | 0.60–0.77 | −50% |
| COVID era | 60 | 0.69 | 0.64–0.74 | −21% |
| Night (0–6) | 100 | 0.67 | 0.62–0.73 | −36% |
| Sensor match needs review | 45 | 0.65 | 0.58–0.75 | −39% |
| Public holidays | 100 | 0.64 | 0.59–0.72 | −31% |

The bias column shows two different failures. Quiet sensors are over-predicted (+36%): the model pulls every place toward a typical volume, so the quietest streets come out too busy. Low-amenity sensors are under-predicted (−50%) because the group mixes quiet streets with busy through-routes that have few shops nearby: Southbank Promenade, Southern Cross Station, Sandridge Bridge, the Convention Centre. Those routes carry most of the group's pedestrians, and amenity counts don't capture what draws people there.

**Candidates from step 1 that are not worse:**
- **Thin precincts.** Carlton (6 sensors, WAPE 0.43) and Parkville (3, 0.50) have lower error than the CBD (60, 0.56). Kensington (2 sensors, 1.38) and North Melbourne (4, 0.53) have intervals too wide to say either way: 0.35–2.56 and 0.31–1.31.
- **Held-out transfer precincts.** Carlton 0.43, UniMelb edge 0.50, North Melbourne 0.78 (interval 0.37–1.64). None is clearly worse than the training area (0.58). These are the K-fold errors; the formal transfer test, which trains on the CBD only, is part of model evaluation.
- **Rare weather.** Rain of 1 mm or more (0.60), 30 °C or hotter (0.57) and cold hours (0.60) are in line with dry, mild hours.
- **Late starters and recent hours.** Sensors that started in 2019 or later (0.62), the post Metro Tunnel era (0.57) and AFL days (0.58) are not worse.

So being thin in the data, which step 1 measured, doesn't by itself make a group worse for the model. The worse groups share something else: low volume, or a pattern the features don't describe (night, holidays, COVID).

## Step 3: mitigation

Each mitigation is run on the same folds, the same training rows and the same scored hours, so only the mitigation differs (settings in `configs/bias.yaml`, `errors.mitigations`):
- **`sensor_equal`:** every sensor's training rows sum to the same weight, so long-running busy CBD sensors stop dominating the loss.
- **`precinct_to_network`:** each precinct's rows are weighted to its share of the walkable network, the step 1 reference.
- **`calibrated`:** no retraining. Each held-out sensor's baseline predictions are scaled so they add up to its counts over its first 28 days. This is the per-site calibration in CLAUDE.md: the model gives the pattern and a few weeks of local counts give the volume.

| WAPE | Baseline | Equal sensors | Precincts to network | **Calibrated** |
|---|---|---|---|---|
| **Overall** | 0.58 | 0.56 | 0.53 | **0.39** |
| Overall bias | −31% | −28% | −25% | **+1%** |
| Quiet sensors | 0.78 | 0.81 | 0.80 | **0.59** |
| Steep paths (≥5%) | 0.76 | 0.75 | 0.71 | **0.51** |
| Low-amenity sensors | 0.71 | 0.70 | 0.72 | **0.48** |
| Match needs review | 0.65 | 0.64 | 0.60 | **0.42** |
| Public holidays | 0.64 | 0.62 | 0.61 | **0.49** |
| COVID era | 0.69 | 0.69 | 0.67 | 0.67 |
| Night | 0.67 | 0.70 | 0.69 | 0.65 |
| CBD | 0.56 | 0.52 | 0.50 | 0.36 |
| Carlton (held out) | 0.43 | 0.43 | 0.46 | 0.57 |
| North Melbourne (held out) | 0.78 | 0.63 | 0.78 | 0.36 |

**Reweighting: tried and rejected.** Both weightings lowered the average error, by up to 0.05, but almost all of the gain went to groups that were already good: CBD, busy sensors and midday hours. Quiet, low-amenity and night errors stayed the same or rose slightly, so the gap between the best and worst groups widened (on time of day, the disparity ratio fell from 0.82 to 0.70–0.73). The weak groups aren't under-weighted: the features don't tell the model how busy a quiet place is, and more weight on those rows doesn't add that information.

**Calibration: adopted where local counts exist.** It cuts the overall error by a third and removes the overall bias. The place and volume groups improve most: quiet sensors from 0.78 to 0.59, low-amenity from 0.71 to 0.48 and steep paths from 0.76 to 0.51. Calibrated forecasts are the right default wherever a few weeks of counts are available.

**Not fixed:**
- **Night and COVID.** These are errors in the pattern over the day and over the years, not in the level, so scaling can't fix them. Because calibration improves daytime hours far more, the relative gap on time of day widens (disparity ratio 0.82 to 0.53), and the same happens on era (0.82 to 0.50). Both are reported separately until the model changes (next steps).
- **Carlton.** Calibration made it worse, 0.43 to 0.57, and turned a small under-prediction into a 34% over-prediction. Its sensors' first 28 days don't look like their later years, so a calibration window has to be chosen with care: a short count taken in an unusual month calibrates every later forecast wrongly.

## Mitigation plan and outcome

Arshmehar's plan, written before step 2, with what was found and done:

| If the model is worse on... | Planned mitigation | Outcome |
|---|---|---|
| Thin precincts and held-out precincts | Weight training rows so each precinct or sensor counts equally; report held-out precincts separately, with sensor counts and intervals | Not worse than overall. Both weightings were tried and rejected (above). Held-out precincts are reported separately in `slice_errors.csv`. |
| Quiet sensors | Relative error; weight sensors equally | Worse (0.78). Relative error is used throughout. Equal sensor weights didn't help (0.81); calibration did (0.59). |
| Night | Relative error, and night reported separately | Worse (0.67), and not fixed by weights or calibration. Reported separately; needs model changes. |
| Early years, short sensor histories | Weight by sensor-year, or a recent-window model | Not worse: sensors from 2019 on have WAPE 0.62, interval 0.52–0.73. No action. |
| COVID and post Metro Tunnel | Keep the flags; report post Metro Tunnel separately | COVID is worse (0.69) and unfixed; post Metro Tunnel is not worse (0.57). Both reported separately. |
| Rain, heat, DEM sensors | Report separately; weight up if much worse | Rain and heat not worse. The 2 DEM sensors are too few to judge. No action. |
| Missing years | Bring the Wayback fill into the pipeline | Still to do (data fix). |
| Sensors with no location | Locations for the 18 retired sensors from the October 2022 sensor table | Still to do (data fix). Until then they're not evaluated. |
| Incomplete sensors and flagged matches | Report sensors missing 20% or more separately; hand-check the 45 flagged matches | Flagged matches are worse (0.65), so the hand check now matters for accuracy too. Calibration brings them to 0.42. The 5 incomplete sensors scored are too few to judge (interval 0.45–5.10). |
| Outage zeros | Already dropped | Dropped from training and scoring. |

## Trade-offs

- **Average against gap.** Reweighting bought up to 0.05 of overall WAPE and paid for it with a wider gap between groups. Since the point of the step is the weak groups, the trade isn't worth it.
- **Calibration needs counts.** It works only where a short local count exists. Segments that have never been counted keep the uncalibrated error, which is worst on quiet, low-amenity streets. Any forecast for an uncounted place should say it is uncalibrated.
- **Calibration widens the time gap.** It helps daytime far more than night, so night's relative disadvantage grows even though its error doesn't get worse.
- **The calibration window matters.** Four weeks that aren't typical (Carlton) make every later forecast worse. A longer or more recent window would be safer, at the cost of needing more counts.
- **Relative error.** WAPE makes night and quiet streets visible but understates busy-hour errors in people terms, so MAE is reported beside it (`mae` in `slice_errors.csv`).
- **Small groups.** Groups of 2–6 sensors have wide intervals. They're reported, not optimised for: tuning a model to 2 sensors would overfit them.
- **The Wayback fill**, once added, brings about two years from an archived copy of the City's own export, which matched the archive on about 98.5% of overlapping hours. The `source` slice keeps it checkable.

## Next steps

1. **Rerun on the real flow model (step 4).** `slice_errors()` takes any model's out-of-fold predictions, so the flow model's evaluation can produce the same table.
2. **Night and COVID.** Try a lockdown-stringency feature for COVID, and either a separate low-count model or a log-scale loss for night, then rerun step 3.
3. **Calibration window.** Compare 4 weeks, 8 weeks and the most recent 4 weeks, and pick the default for the per-site calibration.
4. **Hand-check the 45 flagged sensor matches.** They're worse in step 2, so wrong segments are costing accuracy.
5. **Data fixes:** bringing the Wayback fill and the retired sensors' locations into the pipeline (Arshmehar's and Shubham's work) adds the 2023–24 rows and puts the 18 sensors into the evaluation.
