"""Observed crowd egress after events, from the per-minute counts.

For each event at an area's venue, the counts at the area's sensors in 5-minute blocks
and by direction over the egress window, plus a summary per night and sensor: total,
peak block, peak count and how long after the lead sensor's peak each sensor peaks.
These are the series the walkers and the microsimulation are compared with.

    .venv/bin/python -m src.simulation.observed --area melbourne_park

Times are Melbourne local time. Events have no published end, so the window is set in
hours after the advertised start (configs/simulation.yaml).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "simulation.yaml"
TZ = "Australia/Melbourne"
COUNTS = ["direction_1", "direction_2", "total"]

log = logging.getLogger("stoa.simulation.observed")


def load_settings(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def event_table(events_yaml: dict[str, Any]) -> pd.DataFrame:
    """Every listed event (name, kind, venue, local start) from configs/events.yaml."""
    rows = []
    for value in events_yaml.values():
        if isinstance(value, list):
            rows += [e for e in value if isinstance(e, dict) and "venue" in e]
    events = pd.DataFrame(rows, columns=["name", "kind", "venue", "start"])
    events["start"] = pd.to_datetime(events["start"]).dt.tz_localize(TZ)
    return events.sort_values("start").reset_index(drop=True)


def area_events(events: pd.DataFrame, area: dict[str, Any]) -> pd.DataFrame:
    """The area's events, each flagged with how many other events nearby that night."""
    chosen = events[
        (events["venue"] == area["event_venue"])
        & events["kind"].isin(area["event_kinds"])
    ].copy()
    nearby = events[events["venue"].isin(area["nearby_venues"])]
    nights = nearby["start"].dt.date.value_counts()
    chosen["night"] = chosen["start"].dt.date
    chosen["other_events_that_night"] = chosen["night"].map(nights).fillna(1) - 1
    return chosen.reset_index(drop=True)


def minute_counts(feed: pd.DataFrame, sensors: list[int]) -> pd.DataFrame:
    """Counts per sensor and minute in local time, keeping each row's latest poll."""
    feed = feed[feed["location_id"].isin(sensors)]
    feed = feed.sort_values("polled_at").drop_duplicates(
        ["location_id", "sensing_datetime"], keep="last"
    )
    ts = pd.to_datetime(feed["sensing_datetime"], utc=True).dt.tz_convert(TZ)
    return pd.DataFrame(
        {
            "location_id": feed["location_id"].to_numpy(),
            "ts_local": ts.reset_index(drop=True),
            "direction_1": feed["direction_1"].to_numpy(),
            "direction_2": feed["direction_2"].to_numpy(),
            "total": feed["total_of_directions"].to_numpy(),
        }
    )


def blocks(counts: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Counts summed into blocks of `minutes`, per sensor."""
    out = counts.assign(block=counts["ts_local"].dt.floor(f"{minutes}min"))
    return out.groupby(["location_id", "block"], as_index=False)[COUNTS].sum()


def egress_series(
    counts: pd.DataFrame, events: pd.DataFrame, area: dict[str, Any]
) -> pd.DataFrame:
    """Each event's block counts at each sensor over the egress window."""
    parts = []
    for _, event in events.iterrows():
        start = event["start"] + pd.Timedelta(hours=area["window_from_hours"])
        end = event["start"] + pd.Timedelta(hours=area["window_to_hours"])
        window = counts[(counts["block"] >= start) & (counts["block"] < end)]
        if window.empty:
            continue
        part = window.assign(
            night=event["night"],
            event=event["name"],
            minutes_after_start=(window["block"] - event["start"]).dt.total_seconds()
            / 60,
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(
            columns=["night", "event", "location_id", "block", "minutes_after_start"]
            + COUNTS
        )
    series = pd.concat(parts, ignore_index=True)
    order = ["night", "event", "location_id", "block", "minutes_after_start"] + COUNTS
    return series[order].sort_values(["night", "location_id", "block"])


def summarise(series: pd.DataFrame, lead_sensor: int) -> pd.DataFrame:
    """Per night and sensor: blocks, total, peak block, peak count and peak delay."""
    rows = []
    keys = ["night", "event", "location_id"]
    for (night, event, sensor), s in series.groupby(keys):
        peak = s.loc[s["total"].idxmax()]
        rows.append(
            {
                "night": night,
                "event": event,
                "location_id": sensor,
                "blocks": len(s),
                "total": int(s["total"].sum()),
                "peak_block": peak["block"],
                "peak_count": int(peak["total"]),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    is_lead = summary["location_id"] == lead_sensor
    lead = summary[is_lead].set_index("night")["peak_block"]
    delay = summary["peak_block"] - summary["night"].map(lead)
    summary["peak_delay_min"] = delay.dt.total_seconds() / 60
    summary.loc[summary["night"].map(lead).isna(), "peak_delay_min"] = np.nan
    return summary


def read_feed(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("minute_feed_*.parquet"))
    if not files:
        raise SystemExit(f"no minute_feed files in {folder}: run dvc pull")
    log.info("reading %d minute files", len(files))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run(area_name: str, settings: dict[str, Any] | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    area = settings["areas"][area_name]
    sensors = [s for s in area["sensors"] if s not in area.get("exclude_sensors", [])]
    events_yaml = yaml.safe_load((REPO_ROOT / settings["events_config"]).read_text())
    events = area_events(event_table(events_yaml), area)
    feed = read_feed(REPO_ROOT / settings["minute_feed_dir"])
    counts = blocks(minute_counts(feed, sensors), area["block_minutes"])
    series = egress_series(counts, events, area)
    summary = summarise(series, area["lead_sensor"])
    if not summary.empty:
        flags = events.set_index("night")["other_events_that_night"]
        summary["other_events_that_night"] = summary["night"].map(flags)

    out = REPO_ROOT / settings["output_dir"] / area_name
    out.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out / "observed_egress.parquet", index=False)
    summary.to_csv(out / "observed_summary.csv", index=False)
    log.info(
        "%s: %d events, %d nights with counts; wrote %s",
        area_name,
        len(events),
        series["night"].nunique(),
        out,
    )
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description="Observed egress after events")
    ap.add_argument("--area", default="melbourne_park")
    args = ap.parse_args()
    summary = run(args.area)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
