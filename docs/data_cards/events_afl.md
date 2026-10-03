# AFL games (event days): Squiggle API

| | |
|---|---|
| **Role** | Event-day features and the separate event-day evaluation split (MCG and Marvel Stadium/Docklands sit in or beside the study area) |
| **Provider** | Squiggle (api.squiggle.com.au), community AFL data API |
| **License** | **Unclear.** No formal licence is published. Fixtures (date, time, venue, teams) are public facts, but treat Squiggle as a convenience source: credit it and don't redistribute bulk dumps. Cross-check against the AFL's official fixtures if needed. |
| **Access** | `python -m src.ingest.events`, one JSON per year: `data/raw/events/afl_games_YYYY.json` |
| **Coverage** | 2009–2026 seasons, 3,664 games across all venues |
| **Melbourne venues** | M.C.G. 873 games; Docklands 744 + "Marvel Stadium" 43 (same venue, renamed). 1,660 games 2009-03-26 → 2026-09-26 at these two. |

## Key fields

`date`, `localtime`, `tz`, `unixtime`, `venue`, `hteam`, `ateam`, `round`, `roundname`, `is_final`, `is_grand_final`, `complete`, scores.

## Known issues

- No attendance figures. Crowd size matters for flow, so attendance needs another source (AFL Tables, footywire). Licensing for those needs checking.
- Venue names vary over time (Docklands = Etihad = Marvel Stadium). Normalise in features.
- AFL only. Other big crowds near the study area are not covered: cricket at the MCG, Australian Open at Melbourne Park, concerts, Moomba, NYE, marathons. Possible sources: CoM `venues-for-event-bookings`, `event-permits-2014-2018` (historical only), and public event calendars.
