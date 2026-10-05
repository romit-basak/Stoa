import re
from datetime import date, timedelta

import pandas as pd
import pytest
import yaml

from src.ingest.calendars import site_holidays
from src.ingest.common import REPO_ROOT

CAL = REPO_ROOT / "configs" / "calendars"
VIC_DATES = REPO_ROOT / "data" / "raw" / "calendars" / "vic_important_dates.csv"
TERM_ROW = re.compile(r"Term (\d) (\d{4}) ?- ?(Start|End) date", re.IGNORECASE)
# Known errors in the Vic Important Dates CSV, checked against the official term-dates page.
VIC_DATES_ERRATA = {
    # Year mislabelled: 2021-09-17 is the end of term 3 2021.
    "Term 3 2020 - End date",
    # Scheduled date; term 1 2020 actually ended early on 24 March (COVID).
    "Term 1 2020 - End date",
}


def _vic_dates() -> pd.DataFrame:
    if not VIC_DATES.exists():
        pytest.skip("run `python -m src.ingest.calendars` first")
    d = pd.read_csv(VIC_DATES)
    d["date"] = pd.to_datetime(d.important_date, dayfirst=True).dt.date
    return d


@pytest.mark.parametrize(
    "name", ["vic_school_terms", "unimelb_teaching_periods", "neu_academic_calendar"]
)
def test_calendar_ranges_are_ordered(name):
    cal = yaml.safe_load((CAL / f"{name}.yaml").read_text())

    def ranges(node):
        if isinstance(node, dict):
            if "start" in node and "end" in node:
                yield node["start"], node["end"]
            for v in node.values():
                yield from ranges(v)
        elif isinstance(node, list):
            if len(node) == 2 and all(isinstance(x, date) for x in node):
                yield node[0], node[1]
            else:
                for v in node:
                    yield from ranges(v)

    found = list(ranges(cal))
    assert found
    assert all(a <= b for a, b in found)


def test_school_terms_match_vic_important_dates():
    terms = yaml.safe_load((CAL / "vic_school_terms.yaml").read_text())["terms"]
    rows = _vic_dates().query("dateType == 'SCHOOL_TERM'")
    checked = 0
    for name, day in zip(rows.name, rows.date):
        m = TERM_ROW.search(name)
        if not m or int(m[2]) not in terms:
            continue
        if name in VIC_DATES_ERRATA:
            continue
        term = terms[int(m[2])][int(m[1]) - 1]
        if m[3].lower() == "start":
            # The CSV gives the teachers' start in some years and the students' in others.
            assert day in {term["start"], term.get("students_start")}, name
        else:
            assert term["end"] == day, name
        checked += 1
    assert checked >= 60  # 2019-2026: 8 years x 4 terms x start/end


def test_holidays_package_matches_vic_important_dates():
    rows = _vic_dates().query("dateType == 'PUBLIC_HOLIDAY'")
    official = {d for d in rows.date if 2019 <= d.year <= 2027}
    ours = {
        date.fromisoformat(r["date"]) for r in site_holidays("AU", "VIC", 2019, 2027)
    }
    # The package lists the day off: a holiday on a weekend appears on its substitute weekday.
    missing = {
        d
        for d in official - ours
        if not (d.weekday() >= 5 and d + timedelta(days=7 - d.weekday()) in ours)
    }
    assert missing == set()
