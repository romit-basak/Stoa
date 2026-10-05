# AFL games (event days): Squiggle API

| | |
|---|---|
| **Role** | Event-day features and the separate event-day evaluation split (MCG and Marvel Stadium/Docklands sit in or beside the study area) |
| **Provider** | Squiggle (api.squiggle.com.au), community AFL data API |
| **License** | **Unclear.** No formal licence is published. Fixtures (date, time, venue, teams) are public facts, but treat Squiggle as a convenience source: credit it and don't redistribute bulk dumps. Cross-check against the AFL's official fixtures if needed. |
| **Access** | `python -m src.ingest.events`, one JSON per year: `data/raw/events/afl_games_YYYY.json` |
| **Coverage** | 2009–2026 seasons, 3,664 games across all venues |
| **Melbourne venues** | M.C.G. 873 games; Docklands 744 + "Marvel Stadium" 43 (same venue, renamed). 1,660 games 2009-03-26 → 2026-09-26 at these two. |

## Sensor coverage near the venues (event-day evaluation)

- **Marvel Stadium / Docklands:** 7 outdoor sensors within 500 m (25 within 1 km), reporting on every game day in both eras. **The AFL event-day evaluation rests here.**
- **MCG:** no sensor within 500 m. The four within 1 km (IDs 75, 123, 124, 136) include the two Birrarung Marr bridge sensors (45–70% complete). In the archive only 93 of 655 MCG game days have one of them reporting (all 101 in the live window). Treat MCG results as weak evidence.

## Key fields

`date`, `localtime`, `tz`, `unixtime`, `venue`, `hteam`, `ateam`, `round`, `roundname`, `is_final`, `is_grand_final`, `complete`, scores.

## Known issues

- Squiggle has no attendance figures; those come from AFL Tables (below).
- Venue names vary over time (Docklands = Etihad = Marvel Stadium). Normalise in features.
- AFL only. Other big crowds near the study area are not covered: cricket at the MCG, Australian Open at Melbourne Park, concerts, Moomba, NYE, marathons. Possible sources: CoM `venues-for-event-bookings`, `event-permits-2014-2018` (historical only), and public event calendars.

## Attendance (AFL Tables)

| | |
|---|---|
| **Access** | `python -m src.ingest.afl_attendance` → `data/raw/events/afltables_YYYY.html` (raw season pages) + `afl_attendance.csv` (parsed) |
| **Source** | `https://afltables.com/afl/seas/{year}.html`. Pulled 2026-10-04, 2 s between requests. |
| **License** | **None published**, and no `robots.txt` (404). Crowd figures are facts. Credit AFL Tables; don't redistribute the pages. Footywire disallows these pages in robots.txt, so don't use it. Kaggle `stoney71/aflstats` (labelled ODbL, 2012–2025) is a scrape of AFL Tables/Footywire, so its label doesn't clear upstream rights. Use it only as a cross-check. |
| **Fields** | `date`, `local_time`, `melbourne_time`, `attendance`, `venue`, `season` |

**Verification:**
- 3,664 games for 2009–2026, the same total as Squiggle. 1,660 at the MCG (873) or Docklands (787).
- Every Squiggle MCG/Docklands game matches on (date, venue).
- 49 MCG/Docklands games have no attendance: 2020 (20) and 2021 (29), when COVID crowd limits applied. Treat them as unknown crowd, not zero.
- Round-total lines (`Tot Att:`) are excluded by the parser.

## Other events after 2018

The CoM permits dataset ends in 2018. Later major events are easy to look up and cite by hand: fixed-date ones (NYE) need no source at all, and the rest (White Night / Now or Never, Moomba, Australian Open, marathon, Grand Final parade) have official pages or AFL Tables. Keep them in a small hand-curated `events.yaml` with a source URL on each row.
