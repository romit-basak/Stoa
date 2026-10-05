"""AFL match attendance (crowd size) from AFL Tables season pages, 2009 onward.

Usage: python -m src.ingest.afl_attendance

Squiggle (src.ingest.events) has fixtures but no crowd figures. AFL Tables publishes no licence
and no robots.txt; attendance figures are facts. Fetch politely (one page per season, a pause
between requests), keep the raw HTML, and credit AFL Tables. See docs/data_cards/events_afl.md.
"""

from __future__ import annotations

import csv
import re
import time
from datetime import UTC, datetime

from src.ingest.common import download, load_config, raw_dir, record

SOURCE = "events"
GAME = re.compile(
    r"(?P<dow>Mon|Tue|Wed|Thu|Fri|Sat|Sun) (?P<date>\d\d-\w{3}-\d{4}) "
    r"(?P<time>\d{1,2}:\d\d [AP]M)(?: \((?P<melb_time>\d{1,2}:\d\d [AP]M)\))?"
    r"\s*(?:<b>Att: </b>(?P<att>[\d,]+))?\s*<b>Venue:</b>\s*<a[^>]*>(?P<venue>[^<]+)</a>"
)
FIELDS = ["date", "local_time", "melbourne_time", "attendance", "venue", "season"]


def parse_season(html: str, season: int) -> list[dict[str, str]]:
    rows = []
    for m in GAME.finditer(html):
        rows.append(
            {
                # Calendar date as printed (venue-local); no time zone involved.
                "date": datetime.strptime(m["date"], "%d-%b-%Y").date().isoformat(),  # noqa: DTZ007
                "local_time": m["time"],
                "melbourne_time": m["melb_time"] or m["time"],
                "attendance": (m["att"] or "").replace(",", ""),
                "venue": m["venue"].strip(),
                "season": str(season),
            }
        )
    return rows


def main() -> None:
    cfg = load_config()
    ev = cfg["events"]
    out = raw_dir(cfg, SOURCE)
    rows: list[dict[str, str]] = []
    for season in range(ev["start_year"], datetime.now(UTC).year + 1):
        url = ev["afltables_season_url"].format(year=season)
        page = out / f"afltables_{season}.html"
        download(url, page)
        record(out, page, url, license="no licence stated (facts); credit AFL Tables")
        rows += parse_season(page.read_text(encoding="latin-1"), season)
        time.sleep(ev["polite_delay_s"])

    dest = out / "afl_attendance.csv"
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    record(
        out,
        dest,
        ev["afltables_season_url"],
        derived_from="afltables_YYYY.html",
        games=len(rows),
        license="no licence stated (facts); credit AFL Tables",
    )


if __name__ == "__main__":
    main()
