"""Calendars: public holidays per site, plus the Vic Government Important Dates dataset.

Usage: python -m src.ingest.calendars

School terms and university teaching periods are hand-maintained facts in configs/calendars/
(the University of Melbourne's website terms forbid scraping). See docs/data_cards/calendars.md.
"""

from __future__ import annotations

import csv
from importlib.metadata import version

import holidays

from src.ingest.common import REPO_ROOT, download, load_config, raw_dir, record

SOURCE = "calendars"


def site_holidays(
    country: str, subdiv: str, first: int, last: int
) -> list[dict[str, str]]:
    days = holidays.country_holidays(
        country, subdiv=subdiv, years=range(first, last + 1)
    )
    return [
        {"date": d.isoformat(), "name": name}
        for d, name in sorted(days.items())
        # Commemorative only, not a day off.
        if "Susan B. Anthony" not in name
    ]


def main() -> None:
    cfg = load_config()
    cal = cfg["calendars"]
    out = raw_dir(cfg, SOURCE)
    first, last = cal["years"]

    for site, spec in cal["holidays"].items():
        dest = out / f"{site}_public_holidays.csv"
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "name"])
            w.writeheader()
            w.writerows(site_holidays(spec["country"], spec["subdiv"], first, last))
        record(
            out,
            dest,
            "python package `holidays`",
            holidays_version=version("holidays"),
            country=spec["country"],
            subdiv=spec["subdiv"],
            years=[first, last],
            license="MIT (package); holiday dates are public facts",
        )

    dest = out / "vic_important_dates.csv"
    download(cal["vic_important_dates"], dest)
    record(out, dest, cal["vic_important_dates"], license="CC-BY-4.0")

    if not (REPO_ROOT / "configs" / "calendars").is_dir():
        raise SystemExit("configs/calendars/ is missing (hand-maintained term dates)")


if __name__ == "__main__":
    main()
