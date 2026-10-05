# NYC DOT pedestrian counts (cross-city transfer test)

| | |
|---|---|
| **Role** | **Evaluation only.** Cross-city transfer test: train on Melbourne, predict NYC. Never training data, or the test stops being a test. |
| **Provider** | NYC Department of Transportation via NYC Open Data (Socrata) |
| **License** | NYC Open Data Law (Admin Code §23-502(d)): public data sets are published "without any registration requirement, license requirement or restrictions on their use"; the source and version must be named and changes described if required. Credit "NYC Department of Transportation, via NYC Open Data". |
| **Access** | `python -m src.ingest.nyc` → `data/raw/nyc/` (pulled 2026-10-04) |

## Files (`data/raw/nyc/`)

| File | Rows | Source |
|---|---|---|
| `bi_annual_pedestrian_counts.geojson` | 114 locations × 111 count columns | `cqsj-cfgu` (primary) |
| `automated_pedestrian_counts.csv` | 1,509,828 | `ct66-47at`, filtered to `travelmode='pedestrian'` (secondary) |
| `automated_counter_sensors.csv` | 67 | `6up2-gnw8` (all modes) |
| `bi-annual-ped-count-readme.pdf` | — | NYC DOT methodology |

## Bi-annual counts (primary)

- **Locations:** 114 points: 100 on-street (mostly retail corridors), 13 East River / Harlem River bridges, Hudson River Greenway. All five boroughs; no duplicate points; no missing geometry.
- **Sampling:** May and September (October since 2020; June in 2024). One weekday (07:00–09:00 = `_am`, 16:00–19:00 = `_pm`) and the adjacent Saturday (12:00–14:00 = `_md`). Mid-block screenline, both sides of the street summed.
- **Values are period totals, not hourly.** Divide by 2 or 3 h for a rate only for display; for evaluation, compare the total with the sum of predicted hours over the same window. Exact count dates are not in the data.
- **0 means not collected**, not zero pedestrians. Mask it.
- **Coverage:** May 2007 → May 2026. No rounds between May 2019 and Oct 2020 (COVID). **May 2019 is missing 64 of 114 locations**; Oct 2020 is missing 19–24.
- **Column names are inconsistent** (`may_22_p_m`, `oct24_am`, `june_24_pm`, `may26_*`). Parse with a regex on (month, year, period), not fixed names.
- **Scale:** Oct 2025 weekday medians ≈ 560/h (AM) and 1,230/h (PM); range ≈ 6–6,700/h.

## Automated counts (secondary)

- 15-minute counts, `in`/`out` direction, status `raw` on every row.
- **Only four pedestrian sites**, each recorded under **two sensor IDs with identical counts** (100% equal on overlapping rows). Keep one ID per site:

| Site | Keep | Duplicate | Data |
|---|---|---|---|
| Willis Ave bridge | 300028963 | 300029648 | 2022-09 → 2025-09 (stopped) |
| High Bridge | 300038506 | 300043077 | 2023-12 → 2026-06 (stopped) |
| Emmons Ave | 300038509 | 300043075 | 2023-12 → now |
| Concrete Plant Park | 300040736 | 300043073 | 2024-04 → now |

- All are bridges or waterfront paths, so they're atypical. Use them only to sanity-check predicted hourly profiles, not for the headline transfer result.

## Using it for the transfer test

- Build the NYC walking graph from OSM and snap each of the 114 points to its segment (both sides of the street are summed in the count, so snap to the street, not one sidewalk).
- Only portable features can be computed in NYC (OSM, terrain, land cover, weather, building heights). CoM-only layers (footpath steepness, CLUE, business establishments, canopy polygons) don't exist here, so the cross-city test uses the portable-feature model.
- Weather: Open-Meteo at an NYC centre point; season from physical quantities, not month.
