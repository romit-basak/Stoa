"""Checks on the per-minute counts archive (src/ingest/minute_feed.py), run by validate.

All are warnings, since the archive is extra evidence and not the training table:
- missing days: every Melbourne day since the archive began has a file;
- late polls: each finished day's last minute is no earlier than `min_last_local`. The feed
  keeps a day until a reset at about 02:00 the next morning, so a short last day means no poll
  landed between midnight and the reset (which also costs the next day's first 55 minutes);
- duplicates: two rows for one sensor and minute in the same poll (the portal publishes a few).
  A minute whose value changed between polls is the portal revising it, about a dozen a day;
  the archive keeps both values, the latest is the one used, and revisions are only counted;
- missing blocks: sensors with a 5-minute block of the hour the portal never publishes (eight
  sensors have no minutes :00-:04, seen 27 October 2026). They are left out of the hourly sums
  and the profiles; the check warns when more sensors join them than `max_gappy_sensors`;
- hourly sums: summed minutes per sensor-hour (the latest poll's value of each minute) against
  the hourly table, for the full hours the archive covers. Minutes with no one counted are
  absent from the feed, so a close match also says the missing minutes were zeros.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.features import minute_profiles as mp
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
    minutes: pd.DataFrame,
    hourly: pd.DataFrame,
    cfg: dict[str, Any],
    now: datetime,
    reporting: dict[str, Any] | None = None,
) -> list[Check]:
    """The per-minute checks. `reporting` is the minute_profiles block of configs/features.yaml;
    without it, no sensor is treated as having missing blocks."""
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

    key = ["location_id", "sensing_datetime"]
    poll = ["polled_at"] if "polled_at" in minutes else []
    dup = int(minutes.duplicated([*key, *poll]).sum())
    revised = int(minutes.drop_duplicates([*key, *poll]).duplicated(key).sum())
    checks.append(
        Check(
            "minute feed: one row per sensor and minute",
            WARNING,
            dup <= cfg["max_duplicates"],
            f"{dup} duplicated within a poll; {revised} revised by a later poll",
            {"duplicates": dup, "revised": revised},
        )
    )

    m = mp.archive_minutes(minutes)
    gappy: list[int] = []
    if reporting is not None:
        sensors = mp.sensor_reporting(m, reporting)
        gappy = sensors.loc[~sensors["usable"], "location_id"].astype(int).tolist()
        blocks = dict(zip(sensors["location_id"], sensors["missing_blocks"]))
        checks.append(
            Check(
                "minute feed: sensors publish every 5-minute block",
                WARNING,
                len(gappy) <= cfg["max_gappy_sensors"],
                f"{len(gappy)} sensors never publish some blocks (left out of the sums "
                f"and profiles): " + ", ".join(f"{s} ({blocks[s]})" for s in gappy)
                if gappy
                else "",
                {"sensors": gappy},
            )
        )

    m = m[~m["location_id"].isin(gappy)]
    sums = m.groupby(["location_id", "ts_local"])["total_of_directions"].sum()
    hour = sums.index.get_level_values("ts_local")
    # Only full, finished hours on both sides: not the archive's first hour if it began
    # part-way through, nor its newest, which may still be filling, nor the hourly table's
    # newest, which the portal publishes before the hour is complete.
    first = mp.first_full_hour(m) if len(m) else hour.min()
    newest = min(hour.max(), hourly["ts_local"].max()) if len(hourly) else hour.max()
    sums = sums[(hour >= first) & (hour < newest)].rename_axis(["location_id", "hour"])
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
            + (f" (without {len(gappy)} sensors with missing blocks)" if gappy else "")
            if len(both)
            else "no overlapping hours yet",
            {"compared": len(both), "mismatch_share": round(share, 4)},
        )
    )
    return checks
