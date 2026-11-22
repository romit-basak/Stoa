# Cross-city transfer test: Melbourne to New York

This is the first test of the flow model outside Melbourne. The portable model, trained only on Melbourne, predicts NYC DOT's bi-annual pedestrian counts, and its predictions are scored against them. NYC's counts are evaluation only: nothing here fits a model on them (CLAUDE.md, "Generalization"; [data card](../data_cards/nyc_dot_pedestrian_counts.md)).

Run on 21 November 2026 with `stoa-flow-portable` v2 (LightGBM, `stoa_train` run `ci__20261116T043438`, MLflow model `m-eaa0f48c…`). Code: `src/models/flow/transfer_nyc.py`, settings in `configs/transfer.yaml`, outputs in [nyc_transfer/](nyc_transfer/).

## Result

On the 97 street locations with counts since 2022 (9 rounds, 2,615 location-window totals):

| | Portable model | Space Syntax | Betweenness |
|---|---|---|---|
| Rank correlation across locations (Spearman, mean of the three windows) | **0.36** | 0.21 | -0.12 |
| Weekday 07-09 / 16-19 / Saturday 12-14 | 0.31 / 0.38 / 0.39 | 0.19 / 0.21 / 0.23 | -0.12 / -0.13 / -0.12 |
| MAE on log totals, uncalibrated | 0.865 | 0.852 | 0.796 |
| Median log ratio (predicted against observed) | -0.46 | -0.39 | -0.26 |
| MAE on log totals after one city-wide scale from 5 locations | 0.823 | 0.831 | 0.812 |
| Share of totals inside the summed 80% bounds | 0.86 | | |

Against the team's goals (`docs/PROJECT_CONTEXT.md`, section 13):
- **Rank correlation of at least 0.6, uncalibrated: missed.** 0.36 on streets. The model ranks NYC's streets better than either baseline, but not well.
- **Error on log totals halved after calibrating on 5 locations: missed.** A city-wide scale from 5 locations takes it from 0.865 to 0.823, about 5%.

## What it shows

- **The model compresses New York's range.** It predicts Midtown Manhattan's busiest blocks (Eighth Avenue, West 34th Street, Fifth Avenue) 6 to 11 times too low, and quiet streets in the outer boroughs 3 to 5 times too high. Across locations, how far off the model is correlates with the observed volume at -0.81, and its log totals vary about 60% as much as the observed ones (standard deviation 0.65 against 1.06). This isn't a range Melbourne never saw: per sidewalk, Midtown's weekday evening hours carry 2,300 to 3,400 people, close to Melbourne's busiest sensors (up to about 2,900 an hour since 2022). The model doesn't recognise those blocks as Melbourne-busy, so what makes a street busy in Melbourne isn't captured in a way the features carry to New York.
- **So calibration helps little.** One city-wide scale removes the overall offset (predictions are about 37% low, a log ratio of -0.46), but not the spread, which is most of the error. More locations help slowly: the calibrated error is 0.920 with 1 location, 0.844 with 3, 0.823 with 5, 0.814 with 10 and 0.807 with 20. A scale plus a slope on log predictions, fitted on the same locations, does no better (0.841 with 5, 0.798 with 20), because the ranking itself is weak.
- **The learned model adds over network structure alone.** Space Syntax, fitted on Melbourne the same way it is in training, ranks NYC's streets at 0.21, and betweenness ranks them backwards (-0.12). So the model's extra ranking must come from its other features: amenities, transit stops, land cover, buildings and the hourly features.
- **Low absolute error isn't a good sign here.** Betweenness has the lowest uncalibrated log error (0.796) while ranking streets backwards. Its predictions barely vary between locations (a standard deviation of 0.41 in log totals, against 1.06 observed), so its errors stay moderate. That's why rank correlation is the headline measure for this test, as the data card says.
- **Including the bridges and greenway flatters the result.** On all 113 locations the model reaches 0.51 (Space Syntax 0.42), because bridge paths are far quieter than commercial streets, which makes them easy to rank. The street-only number is the fair one.

## How it was run

1. **Snapping.** NYC DOT counts both sidewalks of a street at a mid-block screenline, while Melbourne's sensors mostly count one footpath. So each count point is snapped to the street it names (the nearest road whose name matches, else the nearest road), and the nearest sidewalk on each side that runs along it. The two sidewalks' predictions are summed. 98 of the 100 street locations have a sidewalk on both sides. The other 16 points (13 bridges, the Hudson River Greenway, the Staten Island ferry walkway and one pedestrian street) use their nearest path once. The snapped segments are in [units.csv](nyc_transfer/units.csv).
2. **Count days.** The exact count dates aren't published: one weekday and the adjacent Saturday in May and September or October. The model predicts every Tuesday to Thursday of the month that isn't a public holiday, and every Saturday, with that day's hourly weather (ERA5 via Open-Meteo, at NYC's centre point). A window's prediction is the sum over its hours, averaged over the candidate days.
3. **Features.** Built as in training (`src/features/counts_hourly.py`), from the NYC graph and segment features built by `stoa_site_build`. Calendar features Melbourne has and NYC doesn't are labelled assumptions:
   - **School term** is set on. Every count month is inside New York City's public school year.
   - **University teaching and exams** are left missing, as in Melbourne's years before its university calendar starts (2009 to 2014). Setting teaching on instead moves the street result from 0.36 to 0.38 ([uni_teaching/](nyc_transfer/uni_teaching/)).
   - **COVID period** is off. The headline rounds are all after Melbourne's COVID period.
   - **Events** are none. NYC has no event log.
   - **Tram stops** are at the cap distance, since NYC has none.
4. **Baselines.** Space Syntax and betweenness are fitted on every Melbourne fold sensor, as the final model is, and predict NYC from its network metrics and Melbourne's hour-of-week profile.
5. **Scores.** Rank correlation uses each location's mean log total over the rounds, per window, so the gap between busier and quieter windows can't inflate it. Calibration fits one city-wide scale (the mean log ratio) on k random locations' totals and scores the other locations, over 500 draws, reporting the median.

## Sensitivity

| Run | Locations | Rank correlation | log MAE | After 5-location scale |
|---|---|---|---|---|
| Headline: streets, rounds since 2022 | 97 | 0.36 | 0.865 | 0.823 |
| University teaching set on | 97 | 0.38 | 0.841 | 0.812 |
| Streets, every round since 2007 (37 rounds) | 98 | 0.33 | 0.931 | 0.865 |
| All locations, rounds since 2022 | 113 | 0.51 | 0.896 | 0.869 |

The result doesn't hinge on the assumptions. Older rounds score a little worse: the OSM snapshot is from 2026, and the oldest counts are 19 years earlier.

## Limits

- The count days are estimated, so each prediction averages over 4 to 15 candidate days. Rank correlation, which compares locations, is barely affected. Absolute error carries some of it.
- Summing two sidewalks assumes Melbourne's sensors each count one footpath. About two thirds are matched to sidewalks; the rest sit on other footways, streets or paths. That affects the city-wide scale, which calibration removes, more than the ranking.
- The 80% bounds are hourly bounds summed over a window, which is wider than a true 80% interval on the total, so the 0.86 coverage overstates how well calibrated the intervals are.
- The v2 model's final fit is the same as v1's (the same data and parameters), so the result applies to both.

## What this means for the project

The project's claim (CLAUDE.md, "Generalization") was that the model learns patterns in Melbourne, the within-city test covers unfamiliar areas, the cross-city test covers a new city, and a few local counts close much of the gap. On NYC, the first three parts hold only weakly and the last doesn't: the model transfers its ranking of streets better than network structure alone, but a few local counts don't fix a range much wider than Melbourne's. Under core rule 10, the write-up shouldn't claim the model works in other cities as it stands.

Next steps worth trying:
- Train with features that carry scale across cities: residential and job density near the segment, and transit ridership at nearby stations. The OSM features describe what's there, not how many people use it.
- Calibrate with more than a city-wide scale: per-borough or per-density-tier scales, which a city with a few dozen counts could fit.
- Check the hourly shape on NYC's four automated counters (bridges and waterfront paths, so atypical), which the data card set aside for exactly this.
- Repeat the test for every new champion, logged to MLflow with the model it scored.

## Running it

```
.venv/bin/python -m src.ingest.weather --site nyc     # NYC hourly weather, if not pulled
.venv/bin/python -m src.models.flow.transfer_nyc --model-uri models:/stoa-flow-portable@champion
.venv/bin/python -m src.models.flow.transfer_nyc --model-uri <uri> --uni-teaching    # sensitivity
.venv/bin/python -m src.models.flow.transfer_nyc --model-uri <uri> --since 2007      # every round
```

`--model-uri` also takes a downloaded model directory, and `--weather-dir` a folder of `nyc_openmeteo_<year>.json` files. It needs the NYC graph and features (`stoa_site_build` with `site=nyc`) and takes about a minute. Outputs go to `data/models/transfer/nyc/`: `units.csv`, `totals.csv`, `calibration.csv` and `summary.json`. A model registered before the interval bundle (v1) is scored on its mean predictions only.
