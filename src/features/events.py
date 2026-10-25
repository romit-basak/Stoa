"""Event features: when an event is on near a segment, and how far away its venue is.

Events come from AFL fixtures with crowds (data/raw/events/afl_attendance.csv, the MCG and
Marvel Stadium games) and the hand-kept log configs/events.yaml (cricket, concerts, football,
Halloween; every row sourced). Columns are generic, never one per venue (that would be a
location ID in disguise, CLAUDE.md core rule 9):
- event_pre / event_during / event_post: an event at some venue starts within the next
  `pre_hours`, is on, or ended within the last `post_hours`;
- event_dist_m: walking distance from the segment to the nearest venue with an event in any of
  those phases this hour (the cap when there is none);
- event_day_dist_m: the same for any event that local day, including the ones logged with a
  date only.
Events without an end time last a default per kind (`events.default_hours`, a labelled
assumption); a row's `duration_h` overrides it. Expected crowds aren't used: the log has none,
and AFL Tables' attendance is known only after a game.

Venue distances are per segment (osm_segment_venue_dist.parquet, from the segment stage); the
hourly columns are added by the training frame, which knows each row's segment and hour.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from src.ingest.common import REPO_ROOT

PHASES = ("pre", "during", "post")


def load_log(path=None) -> dict[str, Any]:
    with open(path or REPO_ROOT / "configs" / "events.yaml") as f:
        return yaml.safe_load(f)


def timed_events(
    log: dict[str, Any],
    afl: pd.DataFrame,
    afl_venues: dict[str, str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Events with a start time: venue, kind, start and end (local, naive)."""
    rows = []
    for e in log.get("events", []):
        if not e.get("venue") or "date only" in e.get("time_note", ""):
            continue
        start = pd.Timestamp(e["start"])
        if e.get("end"):
            end = pd.Timestamp(e["end"])
        else:
            hours = e.get("duration_h", cfg["default_hours"][e["kind"]])
            end = start + pd.Timedelta(hours=hours)
        rows.append(
            {"venue": e["venue"], "kind": e["kind"], "start": start, "end": end}
        )
    a = afl[afl["venue"].isin(afl_venues)]
    for date, t, v in zip(a["date"], a["local_time"], a["venue"]):
        start = pd.to_datetime(f"{date} {t}", format="%Y-%m-%d %I:%M %p")
        end = start + pd.Timedelta(hours=cfg["default_hours"]["afl"])
        rows.append({"venue": afl_venues[v], "kind": "afl", "start": start, "end": end})
    return pd.DataFrame(rows, columns=["venue", "kind", "start", "end"])


def event_days(log: dict[str, Any], timed: pd.DataFrame) -> pd.DataFrame:
    """(date, venue) for every event at a venue, timed or logged with a date only."""
    rows = [
        {"date": pd.Timestamp(e["start"]).normalize(), "venue": e["venue"]}
        for e in log.get("events", [])
        if e.get("venue")
    ]
    rows += [
        {"date": s.normalize(), "venue": v}
        for s, v in zip(timed["start"], timed["venue"])
    ]
    return pd.DataFrame(rows, columns=["date", "venue"]).drop_duplicates()


def venue_hours(timed: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per (local hour, venue, phase) the venue has an event in."""
    rows = []
    pre, post = (
        pd.Timedelta(hours=cfg["pre_hours"]),
        pd.Timedelta(hours=cfg["post_hours"]),
    )
    for venue, start, end in zip(timed["venue"], timed["start"], timed["end"]):
        spans = {
            "pre": (start - pre, start),
            "during": (start, end),
            "post": (end, end + post),
        }
        for phase, (a, b) in spans.items():
            # Every hour that overlaps [a, b): from the hour containing a to the one before b.
            for h in pd.date_range(
                a.floor("h"), (b - pd.Timedelta(seconds=1)).floor("h"), freq="h"
            ):
                rows.append({"hour": h, "venue": venue, "phase": phase})
    return pd.DataFrame(rows, columns=["hour", "venue", "phase"]).drop_duplicates()


def venue_geometries(pbf, venues: dict[str, Any], crs: int) -> gpd.GeoDataFrame:
    """Each venue's outline (an OSM way) or, for a street, its named road ways."""
    ways = {
        k: int(v["osm"].split("/")[1])
        for k, v in venues.items()
        if v.get("osm", "").startswith("way/")
    }
    parts = []
    if ways:
        quoted = ",".join(f"'{w}'" for w in ways.values())
        polys = pyogrio.read_dataframe(
            pbf,
            layer="multipolygons",
            where=f"osm_way_id IN ({quoted})",
            columns=["osm_way_id"],
        )
        by_id = {int(i): g for i, g in zip(polys["osm_way_id"], polys.geometry)}
        parts += [
            {"venue": k, "geometry": by_id[w]} for k, w in ways.items() if w in by_id
        ]
    for k, v in venues.items():
        if "osm_name" in v:
            lines = pyogrio.read_dataframe(
                pbf, layer="lines", where=f"name = '{v['osm_name']}'", columns=["name"]
            )
            if len(lines):
                parts.append({"venue": k, "geometry": lines.geometry.union_all()})
    if not parts:
        return gpd.GeoDataFrame(columns=["venue"], geometry=gpd.GeoSeries([], crs=crs))
    return gpd.GeoDataFrame(parts, geometry="geometry", crs=4326).to_crs(crs)


def venue_distances(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    venues: gpd.GeoDataFrame,
    csr,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Walking distance from each segment's nearer end to each venue, where within the cap.

    A venue is reached at any graph node within `venue_reach_m` of its outline.
    """
    indptr, nbr, _, wt, eu, ev = csr
    n = len(indptr) - 1
    w = pd.DataFrame({"r": np.repeat(np.arange(n), np.diff(indptr)), "c": nbr, "w": wt})
    w = w.groupby(["r", "c"], sort=False)["w"].min().reset_index()
    g = csr_matrix(
        (w["w"].to_numpy(), (w["r"].to_numpy(), w["c"].to_numpy())), shape=(n, n)
    )
    pos = pd.Series(np.arange(len(nodes)), index=nodes["node_id"])
    cap = float(cfg["max_distance_m"])
    out = []
    for venue, shape in zip(venues["venue"], venues.geometry):
        near = nodes.index[nodes.distance(shape) <= cfg["venue_reach_m"]]
        src = pos[nodes.loc[near, "node_id"]].to_numpy()
        if len(src) == 0:
            continue
        d = dijkstra(g, directed=False, indices=src, min_only=True, limit=cap)
        seg = np.minimum(d[eu], d[ev])
        keep = np.isfinite(seg)
        out.append(
            pd.DataFrame(
                {
                    "segment_id": edges["segment_id"].to_numpy()[keep],
                    "venue": venue,
                    "dist_m": seg[keep].astype("float32"),
                }
            )
        )
    if not out:
        return pd.DataFrame(columns=["segment_id", "venue", "dist_m"])
    return pd.concat(out, ignore_index=True)


def hourly_event_features(
    hour_local: pd.Series,
    segment: pd.Series,
    hours: pd.DataFrame,
    days: pd.DataFrame,
    dist: pd.DataFrame,
    cap: float,
) -> pd.DataFrame:
    """The event columns for rows given by their local hour (naive) and segment."""
    n = len(hour_local)
    out = {f"event_{p}": np.zeros(n, dtype=bool) for p in PHASES}
    out["event_dist_m"] = np.full(n, cap, dtype="float32")
    out["event_day_dist_m"] = np.full(n, cap, dtype="float32")
    h = hour_local.to_numpy()
    day = hour_local.dt.normalize().to_numpy()
    seg = segment.to_numpy()
    for venue in sorted(set(hours["venue"]) | set(days["venue"])):
        d = dist[dist["venue"] == venue].set_index("segment_id")["dist_m"]
        dv = pd.Series(seg).map(d).fillna(cap).to_numpy(dtype="float32")
        vh = hours[hours["venue"] == venue]
        any_phase = np.zeros(n, dtype=bool)
        for p in PHASES:
            on = np.isin(h, vh.loc[vh["phase"] == p, "hour"].to_numpy())
            out[f"event_{p}"] |= on & (dv < cap)
            any_phase |= on
        out["event_dist_m"] = np.where(
            any_phase, np.minimum(out["event_dist_m"], dv), out["event_dist_m"]
        )
        on_day = np.isin(day, days.loc[days["venue"] == venue, "date"].to_numpy())
        out["event_day_dist_m"] = np.where(
            on_day, np.minimum(out["event_day_dist_m"], dv), out["event_day_dist_m"]
        )
    return pd.DataFrame(out, index=hour_local.index)
