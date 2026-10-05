# Calendars: public holidays, school terms, university teaching periods

| Item | Where | Coverage | Source / license |
|---|---|---|---|
| Public holidays per site | `data/raw/calendars/{melbourne,neu_boston,nyc}_public_holidays.csv` via `python -m src.ingest.calendars` | 2007–2027 | Python `holidays` 0.105 (MIT, pinned in `pyproject.toml`): AU/VIC, US/MA, US/NY |
| Vic Government Important Dates | `data/raw/calendars/vic_important_dates.csv` (same command) | Public holidays 2019–2027, school terms 2019–2030 | DataVic, CC BY 4.0. Cross-check only. |
| Victorian school terms | `configs/calendars/vic_school_terms.yaml` (hand-maintained) | 2009–2026 | vic.gov.au term-dates page (CC BY 4.0) for 2013–2026; Wayback copies of the former DEECD page for 2009–2012 |
| University of Melbourne teaching periods | `configs/calendars/unimelb_teaching_periods.yaml` (hand-maintained) | 2015–2026 | Official dates pages, read via Wayback |
| Northeastern academic calendar | `configs/calendars/neu_academic_calendar.yaml` (hand-maintained) | Fall 2026 – Summer 2027 | Registrar PDF (2026-06-04) |

## Verification

`tests/ingest/test_calendars.py` (passes):
- **Public holidays:** every Victorian public holiday in the Important Dates CSV for 2019–2027 is in the `holidays` output. Weekend holidays are matched to their weekday substitute: the package lists the day off (e.g., Australia Day 2020 and 2025 on Monday the 27th), which is what matters for foot traffic. The package also tracks rule changes: Easter Sunday from 2016, AFL Grand Final Friday from 2015 (2020 on 23 Oct), the 2022 National Day of Mourning.
- **School terms:** every 2019–2026 term start/end in the CSV matches the YAML. The CSV's "start date" is the teachers' start in some years and the students' in others, so both are accepted. **Known errors in the CSV:** "Term 3 2020 – End date" is dated 2021-09-17 (wrong year), and "Term 1 2020 – End date" gives the scheduled 27 March, but term 1 2020 ended early on 24 March (COVID).
- **UniMelb dates:** 2015 and 2019 spot-checked by hand against the archived page text. Other years come from the research extract and were not re-read, because the Wayback Machine rate-limited further requests.
- **All YAML ranges:** start ≤ end.

## Rules and caveats

- **Do not scrape unimelb.edu.au.** Its website terms say: "You must not … use automated means to retrieve information from a University website without our permission". The dates are facts, entered by hand with their sources.
- **2020–21:** the terms and semesters exist on paper, but lockdowns and remote learning broke the link to foot traffic. Use a lockdown feature; don't let the term flag carry it.
- **Melbourne Cup** is a metropolitan holiday; fine for the CoM study area.
- **NYC:** the `holidays` NY output drops Susan B. Anthony Day (commemorative, not a day off). NYC public-school calendars are not included.
- **Boston:** Evacuation Day (17 Mar) and Bunker Hill Day (17 Jun) are Suffolk County observances missing from the package; listed in the NEU YAML for the demo.
- **Not collected:** UniMelb dates before 2015 (archived pages exist), and NEU calendars before 2026–27 (PDFs at https://registrar.northeastern.edu/article/past-calendars/). Neither is needed: the UniMelb-edge sensors start in 2015, and NEU is a demonstration for the current year.
