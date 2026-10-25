"""The feature tables' columns: type, role, and whether a feature is portable to other cities.

Usage: python -m src.features.schema --write    # regenerate configs/schemas/*.yaml

One place declares every column of the tables the pipeline builds, so that:
- validation checks the tables against it (configs/schemas/counts_hourly.yaml is generated from
  here; tests/features/test_schema.py fails if the two differ);
- the training frame picks its columns by role, never by hand. Only `feature` columns go into a
  model, so IDs, coordinates and the target can't leak in (CLAUDE.md core rules 2 and 9);
- the portable model uses only `portable` features: ones any city has, from OSM, weather,
  calendars and the sun. Melbourne-only ones come from City of Melbourne layers or Melbourne
  events (CLAUDE.md, "Portable vs. Melbourne-rich model").

Roles:
- key: identifies a row (location_id and ts_utc, segment_id)
- label: the target (pedestriancount)
- feature: may go into a model
- aux: needed to build or join features but not a feature itself (local time stamps, the
  direction counts, which add up to the target, segment bearing)
- meta: provenance, quality flags and descriptions
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import yaml

from src.features.common import load_features_config
from src.ingest.common import REPO_ROOT, load_config

ROLES = ("key", "label", "feature", "aux", "meta")


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str  # the pandas dtype the table is read back with
    role: str
    source: str
    portable: bool | None = None  # set for features only
    description: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"{self.name}: unknown role {self.role}")
        if (self.role == "feature") != (self.portable is not None):
            raise ValueError(
                f"{self.name}: portable is set for features, and only them"
            )


def _f(name, dtype, source, portable, description=""):
    return Column(name, dtype, "feature", source, portable, description)


# --------------------------------------------------------------------------------------------
# counts_hourly: one row per sensor and hour (src/features/counts_hourly.py)
# --------------------------------------------------------------------------------------------

WEATHER = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "is_day",
    "precipitation",
    "rain",
    "shortwave_radiation",
]


def counts_hourly(timezone: str) -> list[Column]:
    c = "counts"
    t = "time"
    s = "sensor_match"
    return [
        Column(
            "location_id", "int32", "key", c, description="City of Melbourne sensor"
        ),
        Column(
            "ts_utc", "datetime64[us, UTC]", "key", c, description="Start of the hour"
        ),
        Column("ts_local", f"datetime64[us, {timezone}]", "aux", t),
        Column("date_local", "datetime64[us]", "aux", t),
        _f("hour_local", "int8", t, True),
        Column("source", "category", "meta", c, description="archive, live or wayback"),
        Column("pedestriancount", "int32", "label", c),
        Column(
            "direction_1",
            "float32",
            "aux",
            c,
            description="Adds up to the target: never a feature",
        ),
        Column(
            "direction_2",
            "float32",
            "aux",
            c,
            description="Adds up to the target: never a feature",
        ),
        Column(
            "suspect_outage_zero",
            "bool",
            "meta",
            c,
            description="Dropped before training",
        ),
        _f("weekday", "int8", t, True),
        Column(
            "hour_of_week",
            "int16",
            "aux",
            t,
            description="weekday x hour; joins open hours",
        ),
        _f("is_weekend", "bool", t, True),
        _f("is_public_holiday", "bool", "calendars", True),
        _f("school_term", "boolean", "calendars", True),
        _f("uni_teaching", "boolean", "calendars", True),
        _f("uni_exams", "boolean", "calendars", True),
        _f("sun_elevation_deg", "float32", "sun", True),
        Column(
            "sun_azimuth_deg",
            "float32",
            "aux",
            "sun",
            description="For the shade lookup; doesn't transfer between hemispheres",
        ),
        _f("covid_period", "bool", t, True, "Lockdown dates are the site's own"),
        _f("post_metro_tunnel", "bool", t, False, "Melbourne's Metro Tunnel opening"),
        *[_f(f"wx_{w}", "float32", "weather", True) for w in WEATHER],
        Column("sensor_description", "str", "meta", s),
        Column("location_type", "str", "meta", s),
        Column(
            "segment_id", "str", "meta", s, description="Joins the segment features"
        ),
        Column("match_method", "str", "meta", s),
        Column("needs_review", "boolean", "meta", s),
        _f("slope_pct", "float32", "footpath_steepness", False, "CoM footpath survey"),
        _f("slope_pct_local_mean", "float32", "footpath_steepness", False),
        _f("elevation_m", "float32", "footpath_steepness", False),
        Column("place_source", "category", "meta", "footpath_steepness"),
        Column("has_location", "bool", "meta", s),
    ]


# --------------------------------------------------------------------------------------------
# osm_segment_features: one row per graph segment (src/features/segment_features.py)
# --------------------------------------------------------------------------------------------

HIGHWAY_CLASSES = [
    "footway",
    "path",
    "pedestrian",
    "steps",
    "corridor",
    "living_street",
    "residential",
    "service",
    "unclassified",
    "tertiary",
    "secondary",
    "primary",
    "trunk",
    "cycleway",
    "track",
    "other",
]
FLAGS = [
    "is_sidewalk",
    "is_crossing",
    "is_steps",
    "is_indoor",
    "is_bridge",
    "is_tunnel",
    "is_covered",
    "is_inferred",
    "is_area_edge",
    "is_detected",
]
SIDEWALK = [
    "sidewalk_both",
    "sidewalk_one_side",
    "sidewalk_none",
    "sidewalk_separate",
    "sidewalk_missing",
]
METRIC = [
    "met_betweenness",
    "met_betweenness_lw",
    "met_closeness",
    "met_harmonic",
    "met_reach",
    "met_reach_len",
    "junctions",
]
ANGULAR = [
    "ang_choice",
    "ang_choice_lw",
    "ang_node_count",
    "ang_reach_len",
    "ang_total_depth",
    "ang_integration",
    "ang_choice_norm",
]


def segment_features(fcfg: dict[str, Any]) -> list[Column]:
    o = "osm"
    cats = list(fcfg["poi_categories"])
    cols = [
        Column(
            "segment_id",
            "str",
            "key",
            o,
            description="<way_id>-<k>, stable within one extract",
        ),
        Column("way_id", "int64", "meta", o),
        Column("name", "str", "meta", o),
        Column("highway", "str", "aux", o, description="One-hot as hw_*"),
        *[_f(f"hw_{h}", "bool", o, True) for h in HIGHWAY_CLASSES],
        *[_f(f, "bool", o, True) for f in FLAGS],
        *[_f(f, "bool", o, True) for f in SIDEWALK],
        _f("surface_paved", "float64", o, True, "1, 0 or missing"),
        _f("surface_paved_missing", "bool", o, True),
        _f("lit", "float64", o, True, "1, 0 or missing"),
        _f("lit_missing", "bool", o, True),
        _f("width_m", "float64", o, True),
        _f("width_missing", "bool", o, True),
        Column(
            "length_m",
            "float64",
            "aux",
            o,
            description="Depends on how the way was split",
        ),
        Column(
            "bearing_deg",
            "float64",
            "aux",
            o,
            description="Axial, 0-180; used for sensor matching",
        ),
    ]
    for r in fcfg["amenities"]["radii_m"]:
        cols += [
            _f(f"poi_{c}_{r}", "float64", o, True, f"Within {r} m on the network")
            for c in cats
        ]
    cols += [
        _f("building_coverage", "float64", o, True),
        _f("building_count", "int64", o, True),
        _f("green_dist_m", "float64", o, True),
        _f("ped_area_dist_m", "float64", o, True),
        _f("green_adjacent", "bool", o, True),
        _f("tram_stop_dist_m", "float64", o, True),
        _f("bus_stop_dist_m", "float64", o, True),
        _f("rail_station_dist_m", "float64", o, True),
        _f(
            "recent_station_dist_m",
            "float64",
            o,
            False,
            "Metro Tunnel stations; the training frame caps it before they opened",
        ),
        _f(
            "wc_tree_share",
            "float64",
            "worldcover",
            True,
            "Tree share, about 50 m around",
        ),
        _f("wc_grass_share", "float64", "worldcover", True),
        _f("wc_built_share", "float64", "worldcover", True),
        _f("dem_slope_mean_pct", "float64", "dem", True, "Grade along it, every 10 m"),
        _f("dem_slope_max_pct", "float64", "dem", True),
        _f("canopy_cover", "float64", "com_canopy", False, "Tree canopy within 25 m"),
        *[
            _f(f"clue_jobs_{r}", "float64", "com_clue", False, f"Jobs within {r} m")
            for r in fcfg["surroundings"]["jobs_radii_m"]
        ],
        *[
            _f(f"businesses_{r}", "float64", "com_business", False)
            for r in fcfg["surroundings"]["business_radii_m"]
        ],
        _f("degree_min", "int64", o, True),
        _f("degree_max", "int64", o, True),
    ]
    for r in fcfg["network_metrics"]["radii_m"]:
        cols += [_f(f"{m}_{r}", "float64", o, True) for m in METRIC + ANGULAR]
    cols.append(
        _f("in_main_component", "bool", o, True, "In the largest connected part")
    )
    cols.append(
        Column(
            "in_metric_graph",
            "bool",
            "meta",
            o,
            description="False for separate-sidewalk roads left out of network metrics",
        )
    )
    cols.append(
        Column(
            "in_feature_mask",
            "bool",
            "meta",
            o,
            description="Near a mask point at a site built only there (NYC); true elsewhere",
        )
    )
    cols.append(Column("geometry", "geometry", "meta", o, description="Segment line"))
    return cols


SENSOR_COLUMNS = [
    Column("location_id", "int64", "key", "sensor_match"),
    Column("sensor_description", "str", "meta", "sensor_match"),
    Column("location_type", "str", "meta", "sensor_match"),
    Column(
        "location_source",
        "str",
        "meta",
        "sensor_match",
        description="current or wayback_2022-10-24",
    ),
    Column("segment_id", "str", "meta", "sensor_match"),
    Column("match_distance_m", "float64", "meta", "sensor_match"),
    Column("match_method", "str", "meta", "sensor_match"),
    Column("needs_review", "bool", "meta", "sensor_match"),
]


def sensor_osm_features(fcfg: dict[str, Any]) -> list[Column]:
    """The sensor match plus every segment column except the segment's own key."""
    skip = {"segment_id", "geometry"}
    return SENSOR_COLUMNS + [c for c in segment_features(fcfg) if c.name not in skip]


# Features the training frame derives per row from per-segment tables and the row's hour
# (src/features/training_frame.py): shade at the hour's sun position, and events nearby.
DERIVED = [
    _f(
        "shade_portable",
        "float32",
        "shade",
        True,
        "Share in building shadow (OSM heights)",
    ),
    _f(
        "shade_rich",
        "float32",
        "shade",
        False,
        "Share in building shadow (city heights)",
    ),
    _f(
        "event_pre",
        "bool",
        "events",
        True,
        "An event near the segment starts within 2 h",
    ),
    _f("event_during", "bool", "events", True, "An event near the segment is on"),
    _f(
        "event_post",
        "bool",
        "events",
        True,
        "An event near the segment ended within 2 h",
    ),
    _f(
        "event_dist_m",
        "float32",
        "events",
        True,
        "Walk to the nearest venue with an event on",
    ),
    _f(
        "event_day_dist_m",
        "float32",
        "events",
        True,
        "Walk to the nearest venue with an event that day",
    ),
]


def tables(site: str | None = None) -> dict[str, list[Column]]:
    """Every table's columns for a site (default: the site the hourly table is built for)."""
    data_cfg, fcfg = load_config(), load_features_config()
    tz = data_cfg["sites"][site or fcfg["counts_hourly"]["site"]]["timezone"]
    return {
        "counts_hourly": counts_hourly(tz),
        "osm_segment_features": segment_features(fcfg),
        "sensor_osm_features": sensor_osm_features(fcfg),
        "derived": DERIVED,
    }


def features(columns: list[Column], portable_only: bool = False) -> list[str]:
    """Names of the columns a model may use; the portable model's only."""
    return [
        c.name
        for c in columns
        if c.role == "feature" and (c.portable or not portable_only)
    ]


def schema_yaml(table: str, columns: list[Column]) -> str:
    """The expected columns and dtypes, in the format validate.check_schema reads."""
    header = (
        f"# Expected columns and pandas dtypes of the {table} table, checked by\n"
        "# src/pipeline/validate.py. Generated from src/features/schema.py with\n"
        "# `python -m src.features.schema --write`; edit schema.py, not this file.\n"
    )
    body = yaml.safe_dump(
        {"columns": {c.name: c.dtype for c in columns}}, sort_keys=False
    )
    return header + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write", action="store_true", help="regenerate configs/schemas/*.yaml"
    )
    args = ap.parse_args()
    t = tables()
    out = REPO_ROOT / "configs" / "schemas" / "counts_hourly.yaml"
    text = schema_yaml("hourly counts", t["counts_hourly"])
    if args.write:
        out.write_text(text)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    else:
        print(text)


if __name__ == "__main__":
    main()
