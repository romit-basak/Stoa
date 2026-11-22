"""The flow model's input features for any segment and hour, built as training builds them.

A model sees its features only through this module at serving time and only through
src/features/training_frame.py at training time, so the two must agree column for column
(tests/api/test_parity.py checks it on real training rows). Wherever possible this module
calls the training code itself rather than re-implementing it:

- segment features (network, amenities, land use): sensor_osm_features in training, which
  is rows of osm_segment_features; here osm_segment_features by segment_id;
- calendar, sun, COVID and Metro Tunnel flags: counts_hourly.time_features, same calendars;
- weather: counts_hourly.weather_for_counts, on the archive in training, the forecast here;
- shade and events: training_frame.shade_columns and event_columns;
- slope and elevation (rich): sensor_steepness per sensor in training, segment_steepness
  per segment here (the sensor's own values where one sits on the segment);
- the Metro Tunnel stations' distance before they opened: training_frame.design_matrix's cap.

Every response says where its inputs came from (CLAUDE.md core rule 3): the weather source,
calendar years the school and university calendars don't cover, and how each segment's slope
was found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import counts_hourly as ch
from src.features import training_frame as tf
from src.features.common import load_features_config
from src.ingest.common import REPO_ROOT, load_config

STEEPNESS = ["slope_pct", "slope_pct_local_mean", "elevation_m"]


class UnknownSegments(LookupError):
    def __init__(self, segments: list[str]) -> None:
        super().__init__(f"unknown segment_id: {', '.join(segments[:10])}")
        self.segments = segments


class BadHours(ValueError):
    pass


@dataclass
class SiteTables:
    """Everything a site's features need that doesn't change between requests."""

    site: str
    data_cfg: dict[str, Any]
    fcfg: dict[str, Any]
    feats: Path
    segments: pd.DataFrame  # segment features, float32, indexed by segment_id
    steepness: pd.DataFrame  # STEEPNESS + place_source, indexed by segment_id
    shade: pd.DataFrame | None
    holidays: pd.Series
    school: tuple[list, set[int]]
    uni: tuple[dict, set[int]]
    center: tuple[float, float]
    timezone: str
    station_cap_m: float
    calendar_years: set[int] = field(default_factory=set)

    @classmethod
    def load(cls, site: str | None = None) -> SiteTables:
        """The tables for `site` (default the training site, counts_hourly.site)."""
        data_cfg, fcfg = load_config(), load_features_config()
        site = site or fcfg["counts_hourly"]["site"]
        tf.check_site(site, data_cfg, fcfg)
        feats = REPO_ROOT / fcfg["processed_dir"] / "features" / site
        wanted = tf.segment_features("rich")
        segments = pd.read_parquet(
            feats / "osm_segment_features.parquet", columns=["segment_id", *wanted]
        ).set_index("segment_id")
        segments = segments[wanted].astype("float32")
        steepness = pd.read_parquet(
            feats / "segment_steepness.parquet",
            columns=["segment_id", *STEEPNESS, "place_source"],
        ).set_index("segment_id")
        shade_path = feats / "osm_segment_shade.parquet"
        raw = REPO_ROOT / data_cfg["raw_dir"]
        cal = REPO_ROOT / "configs" / "calendars"
        school = ch.school_term_ranges(cal / "vic_school_terms.yaml")
        uni = ch.uni_ranges(cal / "unimelb_teaching_periods.yaml")
        s = data_cfg["sites"][site]
        return cls(
            site=site,
            data_cfg=data_cfg,
            fcfg=fcfg,
            feats=feats,
            segments=segments,
            steepness=steepness,
            shade=pd.read_parquet(shade_path) if shade_path.exists() else None,
            holidays=pd.to_datetime(
                pd.read_csv(raw / "calendars" / f"{site}_public_holidays.csv")["date"]
            ),
            school=school,
            uni=uni,
            center=tuple(s["center"]),
            timezone=s["timezone"],
            station_cap_m=float(fcfg["built_form"]["max_distance_m"]),
            calendar_years=school[1] & uni[1],
        )


def localize(hours: list[pd.Timestamp], tz: str) -> pd.DataFrame:
    """ts_local and ts_utc for naive local hours, as counts_hourly.localize does it.

    A clock hour that doesn't exist (the spring-forward hour) is refused rather than dropped.
    """
    naive = pd.Series(pd.to_datetime(hours)).dt.floor("h")
    if naive.dt.tz is not None:
        raise BadHours("give hours as local clock times without a UTC offset")
    out, dropped = ch.localize(pd.DataFrame({"ts_local_naive": naive}), tz)
    if dropped:
        raise BadHours(
            f"{dropped} hour(s) don't exist on the local clock (clocks go forward)"
        )
    return out.reset_index(drop=True)


def grid(segment_ids: list[str], times: pd.DataFrame) -> pd.DataFrame:
    """One row per (segment, hour), segments outermost."""
    n = len(times)
    rows = times.loc[np.tile(np.arange(n), len(segment_ids))].reset_index(drop=True)
    rows.insert(0, "segment_id", np.repeat(segment_ids, n))
    return rows


def build_features(
    tables: SiteTables,
    segment_ids: list[str],
    hours: list[pd.Timestamp],
    weather: pd.DataFrame,
    variant: str = "rich",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(rows, features, sources) for every segment at every hour.

    `weather` is indexed by UTC hour with Open-Meteo's variable names, as
    counts_hourly.read_weather returns it (src/api/weather.py gives the forecast in that
    shape). `rows` holds segment_id, ts_local and ts_utc; `features` the variant's columns
    as float32, in training order.
    """
    unknown = [s for s in dict.fromkeys(segment_ids) if s not in tables.segments.index]
    if unknown:
        raise UnknownSegments(unknown)
    rows = grid(list(segment_ids), localize(hours, tables.timezone))
    return (
        rows,
        features_for_rows(tables, rows, weather, variant),
        sources(tables, rows, weather),
    )


def features_for_rows(
    tables: SiteTables, rows: pd.DataFrame, weather: pd.DataFrame, variant: str
) -> pd.DataFrame:
    """Features for rows with segment_id, ts_local and ts_utc (the parity test calls this
    with training rows)."""
    cfg = tables.fcfg["counts_hourly"]
    time = ch.time_features(
        rows, tables.holidays, tables.school, tables.uni, tables.center, cfg
    )
    wx = ch.weather_for_counts(rows["ts_utc"], weather, cfg["weather_interval_vars"])
    seg = rows["segment_id"]
    steep = tables.steepness.reindex(seg.to_numpy())[STEEPNESS]
    steep.index = rows.index
    shade = tf.shade_columns(
        seg,
        time["sun_elevation_deg"],
        time["sun_azimuth_deg"],
        tables.feats,
        table=tables.shade,
    )
    events = tf.event_columns(
        rows["ts_local"], seg, tables.site, tables.data_cfg, tables.fcfg
    )
    table = pd.concat([time, wx, steep, shade, events], axis=1)

    row_cols = tf.row_features(variant) + tf.derived_features(variant)
    out = table[row_cols].astype("float32")
    seg_cols = tf.segment_features(variant)
    segs = tables.segments.reindex(seg.to_numpy())[seg_cols]
    if "recent_station_dist_m" in segs.columns:
        # As training_frame.design_matrix: before the Metro Tunnel opened, its stations
        # weren't there.
        before = ~time["post_metro_tunnel"].to_numpy(dtype=bool)
        segs.loc[before, "recent_station_dist_m"] = tables.station_cap_m
    segs.index = rows.index
    return pd.concat([out, segs.astype("float32")], axis=1)


def sources(
    tables: SiteTables, rows: pd.DataFrame, weather: pd.DataFrame
) -> dict[str, Any]:
    years = sorted(set(rows["ts_local"].dt.year) - tables.calendar_years)
    place = tables.steepness["place_source"].reindex(rows["segment_id"].unique())
    return {
        "weather": weather.attrs.get("source", "unknown"),
        "calendar_years_not_covered": years,
        "steepness": {k: str(v) for k, v in place.items()},
        "segment_features": "OpenStreetMap (ODbL) and City of Melbourne open data",
    }
