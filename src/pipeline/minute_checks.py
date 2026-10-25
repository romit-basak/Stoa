"""Checks on the per-minute counts archive (src/ingest/minute_feed.py), run by validate.

All are warnings, since the archive is extra evidence and not the training table:
- missing days: every Melbourne day since the archive began has a file;
- late polls: each finished day's last minute is no earlier than `min_last_local`. The feed
  keeps a day until a reset at about 02:00 the next morning, so a short last day means no poll
  landed between midnight and the reset (which also costs the next day's first 55 minutes);
- duplicates: rows sharing a sensor and minute (the portal publishes some);
- hourly sums: summed minutes per sensor-hour against the hourly table, for hours the archive
  covers. Minutes with no one counted are absent from the feed, so a close match also says the
  missing minutes were zeros.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.validate import WARNING, Check

TZ = "Australia/Melbourne"


def read_archive(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("minute_feed_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def missing_days(days: list[date], today: date) -> list[str]:
    if not days:
        return []
    want = pd.date_range(
        min(days), pd.Timestamp(today) - pd.Timedelta(days=1), freq="D"
    )
    have = {pd.Timestamp(d) for d in days}
    return [d.strftime("%Y-%m-%d") for d in want if d not in have]


def minute_checks(
    minutes: pd.DataFrame, hourly: pd.DataFrame, cfg: dict[str, Any], now: datetime
) -> list[Check]:
    if minutes.empty:
        return [
            Check("minute feed: archive held", WARNING, True, "no archive held yet")
        ]
    today = pd.Timestamp(now).tz_convert(TZ).date()
    local = minutes["sensing_datetime"].dt.tz_convert(TZ)
    day = local.dt.date
    days = sorted(set(day))
    checks = []

    gone = missing_days(days, today)
    checks.append(
        Check(
            "minute feed: a file for every day",
            WARNING,
            not gone,
            ", ".join(gone),
            {"missing": gone},
        )
    )

    last = local.groupby(day).max()
    cutoff = pd.Timestamp(cfg["min_last_local"]).time()
    late = [str(d) for d, t in last.items() if d < today and t.time() < cutoff]
    checks.append(
        Check(
            "minute feed: polled late enough each day",
            WARNING,
            not late,
            f"days ending before {cfg['min_last_local']}: {', '.join(late)}"
            if late
            else "",
            {"days": late},
        )
    )

    dup = int(minutes.duplicated(["location_id", "sensing_datetime"]).sum())
    checks.append(
        Check(
            "minute feed: one row per sensor and minute",
            WARNING,
            dup <= cfg["max_duplicates"],
            f"{dup} duplicated",
            {"duplicates": dup},
        )
    )

    hour = local.dt.floor("h")
    sums = (
        minutes.assign(hour=hour)
        .groupby(["location_id", "hour"])["total_of_directions"]
        .sum()
    )
    # Only hours wholly inside the archive: not the newest hour, which may still be filling.
    sums = sums[sums.index.get_level_values("hour") < hour.max()]
    h = hourly.set_index(["location_id", "ts_local"])["pedestriancount"]
    both = pd.concat([sums.rename("minutes"), h.rename("hourly")], axis=1, join="inner")
    off = (both["minutes"] - both["hourly"]).abs() > (
        cfg["sum_tolerance"] * both["hourly"]
    ).clip(lower=2)
    share = float(off.mean()) if len(both) else 0.0
    checks.append(
        Check(
            "minute feed: minutes add up to the hourly counts",
            WARNING,
            share <= cfg["max_mismatch_share"],
            f"{share:.1%} of {len(both)} sensor-hours differ"
            if len(both)
            else "no overlapping hours yet",
            {"compared": len(both), "mismatch_share": round(share, 4)},
        )
    )
    return checks
