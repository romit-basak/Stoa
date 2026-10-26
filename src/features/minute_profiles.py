"""Within-hour profiles from the per-minute counts archive (src/ingest/minute_feed.py).

Usage: python -m src.features.minute_profiles

A data product for the walkers and the microsimulation, so they can start from measured
within-hour shapes instead of an even spread. Nothing here is modelled.

Only sensor-hours that can be trusted are used. The feed leaves out minutes with no one counted,
so a missing minute is either a zero or a missed reading. An hour is kept when its minutes add up
to the hourly table's count (within `sum_tolerance`, or 2 people): then the missing minutes were
zeros. It also needs at least `min_hour_count` people, since a quiet hour's shares are noise.
The archive's newest hour is left out because it may still be filling, and so is the hourly
table's, which the portal publishes before the hour is complete.

Writes data/processed/features/melbourne/:
- minute_hour_status.parquet: every sensor-hour of the hourly table inside the archive's span,
  with whether the archive has it, whether its minutes add up, and whether it was kept. The bias
  step (src/bias/minute_coverage.py) reads it to see which kinds of hours the profiles miss;
- minute_hours.parquet: one row per kept sensor-hour. `peak15` is four times the busiest clock
  quarter's count over the hour's count (1 when the hour is even, 4 when everyone came in one
  quarter); `top_minute_share` is the busiest minute's share. `hour_type` is the event phase at
  the sensor's segment (event_during, event_pre, event_post, from the training frame's event
  columns), else ordinary;
- minute_profiles.parquet: the mean share of the hour in each minute, by sensor, weekday or
  weekend, local hour and hour type, with the number of hours behind each mean;
- minute_profiles_meta.json.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from src.features.common import load_features_config, log, write_meta, write_parquet
from src.ingest.common import REPO_ROOT, load_config, setup_logging

TZ = "Australia/Melbourne"
KEYS = ["location_id", "ts_local"]


def read_archive(folder) -> pd.DataFrame:
    files = sorted(folder.glob("minute_feed_*.parquet"))
    if not files:
        # Typed, so the steps after it work on an empty archive (a clone without bucket
        # credentials, where pull_minute_feed is skipped) as on a full one.
        return pd.DataFrame(
            {
                "location_id": pd.Series(dtype="int32"),
                "sensing_datetime": pd.Series(dtype="datetime64[s, UTC]"),
                "total_of_directions": pd.Series(dtype="int32"),
            }
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def archive_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    """One row per sensor and minute (the latest poll's), with its local hour and minute."""
    m = minutes
    if "polled_at" in m:
        m = m.sort_values("polled_at")
    m = m.drop_duplicates(["location_id", "sensing_datetime"], keep="last")
    utc = m["sensing_datetime"].dt.tz_convert("UTC")
    # Melbourne's offsets are whole hours, so flooring in UTC gives the local hour and never
    # meets the repeated hour when daylight saving ends.
    return m.assign(
        location_id=m["location_id"].astype("int64"),
        ts_local=utc.dt.floor("h").dt.tz_convert(TZ),
        minute=utc.dt.minute.astype("int64"),
    )


def hour_status(
    m: pd.DataFrame, hourly: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Each hourly-table sensor-hour inside the archive's span, and what became of it.

    `m` comes from archive_minutes. Columns: count, minute_sum, in_archive, adds_up, kept.
    """
    cols = [*KEYS, "count", "minute_sum", "in_archive", "adds_up", "kept"]
    if m.empty:
        # Typed, so filtering on the flags and the saved file work with no rows too.
        return pd.DataFrame(
            {
                "location_id": pd.Series(dtype="int64"),
                "ts_local": pd.Series(dtype=f"datetime64[ns, {TZ}]"),
                "count": pd.Series(dtype="int64"),
                "minute_sum": pd.Series(dtype="float64"),
                **{c: pd.Series(dtype=bool) for c in ["in_archive", "adds_up", "kept"]},
            }
        )[cols]
    first, newest = m["ts_local"].min(), m["ts_local"].max()
    # When the archive began mid-hour (25 October 2026 starts at 00:55), that first hour can't
    # add up; it isn't a gap, so it is left out rather than counted against the coverage.
    if m["sensing_datetime"].min() != first:
        first = first + pd.Timedelta(hours=1)
    h = hourly.assign(location_id=hourly["location_id"].astype("int64"))
    # The hourly table's newest hour is left out too: the portal publishes it before the hour
    # is complete, so its count runs short of the minutes until the next pull replaces it.
    if len(h):
        newest = min(newest, h["ts_local"].max())
    h = h[(h["ts_local"] >= first) & (h["ts_local"] < newest)]
    h = h.set_index(KEYS)["pedestriancount"].rename("count")
    sums = m.groupby(KEYS)["total_of_directions"].sum().rename("minute_sum")
    out = h.to_frame().join(sums)
    # The feed publishes no rows for minutes with no one counted, so an hour with a count of 0
    # has nothing in the archive and is still fully covered.
    out.loc[out["count"].eq(0) & out["minute_sum"].isna(), "minute_sum"] = 0
    out["in_archive"] = out["minute_sum"].notna()
    tol = (cfg["sum_tolerance"] * out["count"]).clip(lower=2)
    out["adds_up"] = out["in_archive"] & (
        (out["minute_sum"] - out["count"]).abs() <= tol
    )
    out["kept"] = out["adds_up"] & (out["count"] >= cfg["min_hour_count"])
    return out.reset_index()[cols]


def minute_grid(
    minutes: pd.DataFrame,
    hourly: pd.DataFrame,
    cfg: dict[str, Any],
    which: str = "kept",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sensor-hours (location_id, ts_local, count) and their counts by minute (n x 60).

    `which` is a hour_status flag: "kept" (the default, for the per-sensor profiles) or
    "adds_up" (every hour whose minutes can be trusted, however quiet; for pooled profiles).
    """
    m = archive_minutes(minutes)
    status = hour_status(m, hourly, cfg)
    kept = status[status[which]].set_index(KEYS).sort_index()
    if kept.empty:
        return kept[["count"]].reset_index(), np.zeros((0, 60))
    grid = (
        m.set_index(KEYS)
        .loc[lambda d: d.index.isin(kept.index)]
        .reset_index()
        .pivot_table(
            index=KEYS,
            columns="minute",
            values="total_of_directions",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=kept.index, columns=range(60), fill_value=0)
    )
    return kept[["count"]].reset_index(), grid.to_numpy(dtype=float)


def hour_summary(hours: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """peak15, top_minute_share, hour and day type for each kept sensor-hour."""
    total = grid.sum(axis=1)
    quarters = grid.reshape(len(grid), 4, 15).sum(axis=2)
    out = hours.copy()
    out["peak15"] = (4 * quarters.max(axis=1) / np.maximum(total, 1)).astype("float32")
    out["top_minute_share"] = (grid.max(axis=1) / np.maximum(total, 1)).astype(
        "float32"
    )
    out["hour_local"] = out["ts_local"].dt.hour.astype("int8")
    out["day_type"] = np.where(out["ts_local"].dt.dayofweek >= 5, "weekend", "weekday")
    return out


def hour_types(events: pd.DataFrame) -> np.ndarray:
    """ordinary or the event phase, during first, from the event_* columns."""
    return np.select(
        [events["event_during"], events["event_pre"], events["event_post"]],
        ["event_during", "event_pre", "event_post"],
        "ordinary",
    )


def profiles(summary: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Mean share of the hour in each minute by sensor, day type, local hour and hour type."""
    group = ["location_id", "day_type", "hour_local", "hour_type"]
    if summary.empty:
        return pd.DataFrame(columns=[*group, "minute", "share", "n_hours"])
    shares = grid / np.maximum(grid.sum(axis=1, keepdims=True), 1)
    wide = pd.concat(
        [summary[group].reset_index(drop=True), pd.DataFrame(shares)], axis=1
    )
    g = wide.groupby(group)
    mean = g[list(range(60))].mean()
    n = g.size().rename("n_hours")
    long = mean.stack().rename("share").reset_index()
    long = long.rename(columns={f"level_{len(group)}": "minute"})
    long["minute"] = long["minute"].astype("int8")
    long["share"] = long["share"].astype("float32")
    return long.merge(n.reset_index(), on=group)


def main() -> None:
    setup_logging()
    t = time.perf_counter()
    data_cfg, fcfg = load_config(), load_features_config()
    cfg = fcfg["minute_profiles"]
    site = fcfg["counts_hourly"]["site"]
    folder = REPO_ROOT / data_cfg["raw_dir"] / "melbourne_portal" / "minute_feed"
    out = REPO_ROOT / fcfg["processed_dir"] / "features" / site
    minutes = read_archive(folder)

    if minutes.empty:
        hourly = pd.DataFrame(columns=[*KEYS, "pedestriancount"])
    else:
        start = minutes["sensing_datetime"].min()
        hourly = pd.read_parquet(
            out / "counts_hourly.parquet",
            columns=[*KEYS, "pedestriancount"],
            filters=[("ts_utc", ">=", start.floor("h").tz_convert("UTC"))],
        )
    status = hour_status(archive_minutes(minutes), hourly, cfg)
    hours, grid = minute_grid(minutes, hourly, cfg)
    summary = hour_summary(hours, grid)
    if summary.empty:
        summary["hour_type"] = pd.Series(dtype="object")
    else:
        from src.features.training_frame import event_columns

        seg_of = pd.read_parquet(
            out / "sensor_osm_features.parquet", columns=["location_id", "segment_id"]
        ).set_index("location_id")["segment_id"]
        events = event_columns(
            summary["ts_local"],
            summary["location_id"].map(seg_of),
            site,
            data_cfg,
            fcfg,
        )
        summary["hour_type"] = hour_types(events)
    prof = profiles(summary, grid)

    write_parquet(status, out / "minute_hour_status.parquet")
    write_parquet(summary, out / "minute_hours.parquet")
    write_parquet(prof, out / "minute_profiles.parquet")
    stats = {
        "archive_days": int(
            minutes["sensing_datetime"].dt.tz_convert(TZ).dt.date.nunique()
        )
        if len(minutes)
        else 0,
        "sensor_hours_in_span": len(status),
        "sensor_hours_in_archive": int(status["in_archive"].sum()),
        "sensor_hours_kept": len(summary),
        "sensors": int(summary["location_id"].nunique()),
        "hour_types": summary["hour_type"].value_counts().to_dict(),
        "median_peak15": round(float(summary["peak15"].median()), 3)
        if len(summary)
        else None,
        "runtime_s": round(time.perf_counter() - t, 1),
    }
    write_meta(out / "minute_profiles_meta.json", site=site, **cfg, **stats)
    log.info("minute profiles: %s", stats)


if __name__ == "__main__":
    main()
