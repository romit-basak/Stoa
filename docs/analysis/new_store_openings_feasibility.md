# Feasibility: new-store before/after study (Melbourne)

**Question:** are there large openings close enough to a count sensor, with counts before and after them, to test the amenity what-if (does the model's predicted change match the observed one)?

**Short answer:** not as a statistical study. There is **one strong case (Emporium Melbourne, 2014)** and a few weak ones. That's enough for one or two case studies, plus the Metro Tunnel stations as the main large-attractor test. Keep the study in Tier 2, scoped as case studies.

Reproduce: `python -m src.analysis.new_openings [--sensor-radius 150]` (parameters in `configs/analysis.yaml`). Output: `data/processed/analysis/new_openings_candidates*.csv`.

## Method

1. **Find openings in CLUE** (business establishments and café seating, annual census 2002–2024). A candidate is a property where, between consecutive census years:
   - attractor establishments jumped by ≥ 10 (retail ANZSIC 39–43, food and beverage 45, arts and recreation 89–92), or
   - café/restaurant seats jumped by ≥ 300.
2. **Drop re-coding artefacts:** the net change in attractor establishments within 75 m must be ≥ 5. Otherwise the "jump" is a property ID change, not an opening.
3. **Timing:** an opening first seen in census year Y happened between the Y−1 and Y surveys. Before window = calendar year Y−2; after window = Y+1.
4. **Sensor coverage:** an outdoor sensor within the radius must have ≥ 80% of hours in both windows. That rules out openings whose windows fall in the Nov 2022 – Oct 2024 gap. Windows touching 2020–21 are flagged (COVID), not dropped.

## Results

133 candidate openings, 2 re-coding artefacts.

| Sensor radius | Candidates with a sensor in range | Feasible (counts before and after) | Outside COVID windows |
|---|---|---|---|
| 100 m | 38 | 7 | 6 |
| 150 m | 72 | 15 | 14 |
| 200 m | 96 | 25 | 22 |

**The largest openings mostly have no sensor nearby.** South Wharf (Convention Centre Place, 2011: +151 establishments) and Docklands Drive (2009: +142) have no sensor within 200 m at all. Emporium's two nearest sensors were installed after it opened: 47 at 112 m (2017) and 19 at 147 m (2013).

**At 100 m, none of the seven is a clean test:**

| Census year | Address | Signal | Sensor | Why it's weak |
|---|---|---|---|---|
| 2014 | 427 Docklands Dr (Waterfront City) | +936 seats | 11 (83 m) | Seats only; little change in establishments (+3 nearby) |
| 2016 | 241 Flinders St (Lot 920) | +389 seats | 6 (40 m, Flinders underpass) | Seats only; no net change nearby, so possibly a recount |
| 2018 | 257 Collins St | +300 seats | 53 (80 m) | Seats only |
| 2018 | 360–374 Collins St, Level 12 | +10 establishments | 36 (95 m) | Upper floor: not visible from the street |
| 2021 | 78 Collins St | +12 establishments | 39 (71 m) | Small, in a COVID window |
| 2024 | 37 Swanston St, Level 7 | +11 establishments | 41 (18 m) | Upper-floor studios: not visible from the street |
| 2024 | 121 Therry St (Queen Victoria Market) | +353 seats | 49 (60 m), 51 (81 m) | Seats only; worth a look with news dates |

**The one strong case appears at 150–200 m:**

| Census year | Opening | Signal | Usable sensors |
|---|---|---|---|
| 2014 | 287–321 Lonsdale St (**Emporium Melbourne**) | +193 attractor establishments | 1 Bourke St Mall (North), 3 Melbourne Central (both 164 m), 2 Bourke St Mall (South) (191 m) |

Also at 150–200 m, smaller: 300 La Trobe St (Melbourne Central lower ground, 2012, +36) and 260–274 Collins St (2016, +24).

## Caveats

- **Emporium is ~160–190 m from the sensors and is linked to neighbouring centres by internal bridges,** so part of its traffic never reaches a counted footpath. Expect a diluted signal. Take its exact opening date from news reports (**VERIFY**; CLUE only gives the census year).
- **Seat jumps without establishment gains** may be recounts rather than openings; check against news before use.
- **No openings after 2024:** CLUE ends with the 2024 census, so post-2024 openings (inside the live window, with directional counts) need news reports for both detection and timing.
- **Small openings are invisible** against day-to-day noise; this is why the thresholds are high.

## Recommendation

- **Tier 2, case-study scale:**
  1. Emporium (2014) as the primary case.
  2. One or two news-dated openings near sensors, e.g. at Queen Victoria Market.
- **Method:** difference-in-differences. Compare the change at sensors near the opening with sensors farther away over the same period, so citywide trends (e.g., post-COVID recovery) aren't credited to the store. Then add the opening to the model as a scenario edit and compare predicted with observed change.
  - A match is evidence that amenity what-ifs work.
  - Overprediction is the reverse-causality caveat, measured rather than assumed.
  - Either result is reported.
- **Metro Tunnel stations (opened 2025-11-30)** are the strongest large-attractor test of the same near-vs-far method: sensors within 300 m rose a median ~4.5% year on year while others were flat. They're not stores, so report them as the method check.
