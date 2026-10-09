"""Parse OSM opening_hours strings into a 168-hour weekly mask.

Covers the common forms: "24/7", "Mo-Fr 09:00-17:00; Sa 10:00-14:00", day lists ("Mo,We,Fr"),
wrapping day ranges ("Fr-Mo"), several time ranges ("09:00-12:00,13:00-17:00"), ranges past
midnight ("Fr 18:00-02:00"), open ends ("17:00+", read as open until midnight), "off"/"closed",
and rules without a day (every day). As in the OSM spec, a later rule replaces earlier hours on
the days it names.

Anything else returns None (unknown hours) rather than a guess: months, dates, week numbers,
sunrise/sunset, comments, and rules that only name public or school holidays. PH/SH inside a day
list ("Mo-Fr,PH") is dropped and the rest is kept, since holidays aren't modelled per POI.

Hour h of a day counts as open when the place is open at h:30. Index = weekday * 24 + hour,
with Monday = 0, matching pandas' dayofweek.
"""

from __future__ import annotations

import re

import numpy as np

DAYS = ["mo", "tu", "we", "th", "fr", "sa", "su"]
_DAY = r"(?:mo|tu|we|th|fr|sa|su)"
_TIME = r"\d{1,2}:\d{2}"
_TIME_RANGE = re.compile(rf"^({_TIME})\s*-\s*({_TIME})\+?$|^({_TIME})\+$")
_DAY_SEL = re.compile(
    rf"^(?:{_DAY}(?:\s*-\s*{_DAY})?|ph|sh)(?:\s*,\s*(?:{_DAY}(?:\s*-\s*{_DAY})?|ph|sh))*$"
)
_SEL_ITEM = rf"(?:{_DAY}(?:\s*-\s*{_DAY})?|ph|sh)"
_SEL_PREFIX = re.compile(rf"^({_SEL_ITEM}(?:\s*,\s*{_SEL_ITEM})*)\b\s*(.*)$")
# "Mo-Fr 08:00-18:00, Sa 09:00-12:00" uses a comma where a semicolon belongs.
_COMMA_BEFORE_DAY = re.compile(rf"(?:(?<=\d)|(?<=\+)|(?<=off)),\s*(?={_DAY}\b)")


def _minutes(t: str) -> int | None:
    h, m = t.split(":")
    h, m = int(h), int(m)
    if m >= 60 or h > 48:
        return None
    return h * 60 + m


def _days(sel: str) -> list[int] | None:
    """Day indexes named by a selector like 'Mo-Fr,Su'; [] if it names only holidays."""
    out: list[int] = []
    for part in sel.split(","):
        part = part.strip()
        if part in ("ph", "sh"):
            continue
        if "-" in part:
            a, b = (DAYS.index(x.strip()) for x in part.split("-"))
            out.extend((a + i) % 7 for i in range((b - a) % 7 + 1))
        else:
            out.append(DAYS.index(part))
    return out


def _time_ranges(spec: str) -> list[tuple[int, int]] | None:
    ranges = []
    for part in spec.split(","):
        m = _TIME_RANGE.match(part.strip())
        if not m:
            return None
        if m.group(3):  # "17:00+"
            start, end = _minutes(m.group(3)), 24 * 60
        else:
            start, end = _minutes(m.group(1)), _minutes(m.group(2))
        if start is None or end is None:
            return None
        if end <= start:
            end += 24 * 60  # past midnight
        ranges.append((start, end))
    return ranges


def parse_opening_hours(value: str | None) -> np.ndarray | None:
    """168-element bool mask, or None if the string is missing or outside the supported forms."""
    if not isinstance(value, str):
        return None
    s = value.strip().lower().replace("||", ";")
    if not s or '"' in s:
        return None
    if s == "24/7":
        return np.ones(168, dtype=bool)
    # Each day's own ranges; a later rule replaces them on the days it names. Ranges past
    # midnight spill into the next day when rendered, so "Fr 22:00-02:00; Sa 10:00-14:00" stays
    # open early on Saturday.
    hours: dict[int, list[tuple[int, int]]] = {d: [] for d in range(7)}
    rules = [r.strip() for r in _COMMA_BEFORE_DAY.sub(";", s).split(";") if r.strip()]
    for rule in rules:
        rule = re.sub(r"(?<=[a-z])\s*:\s+", " ", rule)  # "Mo-Fr: 09:00-17:00"
        closed = False
        for word in ("off", "closed"):
            if rule.endswith(word):
                closed, rule = True, rule[: -len(word)].strip()
        m = _SEL_PREFIX.match(rule)
        if m and _DAY_SEL.match(m.group(1).strip()):
            days = _days(m.group(1).strip())
            times = m.group(2).strip()
            if not days:
                if times or closed:
                    continue  # a holiday-only rule: not modelled
                return None
        elif re.match(rf"^{_TIME}", rule) or rule == "24/7" or (closed and not rule):
            days, times = list(range(7)), rule
        else:
            return None
        if closed:
            if times:
                return None
            ranges: list[tuple[int, int]] | None = []
        elif times in ("", "24/7", "00:00-24:00"):
            ranges = [(0, 1440)]
        else:
            ranges = _time_ranges(times)
        if ranges is None:
            return None
        for d in days:
            hours[d] = list(ranges)
    # Minute resolution for the week plus one spill-over day, folded back onto Monday.
    week = np.zeros(8 * 1440, dtype=bool)
    for d, ranges in hours.items():
        for start, end in ranges:
            week[d * 1440 + start : d * 1440 + end] = True
    week[:1440] |= week[7 * 1440 :]
    return week[30 : 7 * 1440 : 60].copy()
