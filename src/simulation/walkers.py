"""Route-sampling walkers for an event's egress (getting_started.md, steps 1 and 2).

For one night at one area:

1. Origins are the graph nodes on segments within a set distance of the venue, and
   destinations are the stations listed in configs/simulation.yaml.
2. Several routes per destination come from shortest paths on segment length, each
   with random noise on the costs, so walkers don't all take one path.
3. Walkers per route and hour are fitted with non-negative least squares so that the
   number passing each fitted sensor matches its observed hourly count, pulled towards
   a prior that splits the crowd by destination weight and favours shorter routes.
4. Each walker's time at the first sensor on its route is drawn at random from the
   event-exit minute profile, and it reaches later sensors at walking speed.
5. Counts per 5-minute block and sensor are split into directions with the sensor's
   hourly direction shares, and compared with the observed series (observed.py).

Sensors left out of the fit are the real test.

    .venv/bin/python -m src.simulation.walkers --area melbourne_park --night 2026-10-27
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.simulation import observed

REPO_ROOT = observed.REPO_ROOT
TZ = observed.TZ

log = logging.getLogger("stoa.simulation.walkers")


@dataclass
class Route:
    destination: str
    segments: list[str]
    offsets_m: list[float]
    length_m: float


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    """Undirected graph on u and v, keeping the shortest segment between two nodes."""
    graph = nx.Graph()
    columns = ["u", "v", "segment_id", "length_m"]
    for u, v, segment, length in edges[columns].itertuples(index=False):
        if graph.has_edge(u, v) and graph[u][v]["length_m"] <= length:
            continue
        graph.add_edge(u, v, segment_id=segment, length_m=float(length))
    return graph


def sample_routes(
    graph: nx.Graph,
    origins: list[Any],
    destinations: dict[str, Any],
    draws: int,
    noise: float,
    seed: int,
) -> list[Route]:
    """Up to `draws` distinct routes per destination, from noisy shortest paths."""
    rng = np.random.default_rng(seed)
    sources = {o for o in origins if o in graph}
    routes = []
    for name, target in destinations.items():
        if target not in graph or not sources:
            log.warning("destination %s is not on the graph", name)
            continue
        seen = set()
        for _ in range(draws):
            factor = dict(zip(graph.edges(), 1.0 + noise * rng.random(graph.size())))

            def cost(u, v, data, factor=factor):
                key = (u, v) if (u, v) in factor else (v, u)
                return data["length_m"] * factor[key]

            try:
                _, path = nx.multi_source_dijkstra(
                    graph, sources, target=target, weight=cost
                )
            except nx.NetworkXNoPath:
                log.warning("no path to %s", name)
                break
            if tuple(path) in seen:
                continue
            seen.add(tuple(path))
            segments, offsets, distance = [], [], 0.0
            for u, v in pairwise(path):
                data = graph[u][v]
                segments.append(data["segment_id"])
                offsets.append(distance + data["length_m"] / 2)
                distance += data["length_m"]
            routes.append(Route(name, segments, offsets, distance))
    return routes


def incidence(
    routes: list[Route], sensor_segments: dict[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Which routes pass each sensor, and how far along each route the sensor is.

    Rows follow the order of `sensor_segments`, columns the order of `routes`.
    """
    sensors = list(sensor_segments)
    passes = np.zeros((len(sensors), len(routes)))
    offset = np.full((len(sensors), len(routes)), np.nan)
    for j, route in enumerate(routes):
        where = dict(zip(route.segments, route.offsets_m))
        for i, sensor in enumerate(sensors):
            if sensor_segments[sensor] in where:
                passes[i, j] = 1.0
                offset[i, j] = where[sensor_segments[sensor]]
    return passes, offset


def route_prior(
    routes: list[Route], weights: dict[str, float], crowd: float, beta: float
) -> np.ndarray:
    """The crowd split by destination weight, then towards each one's shorter routes."""
    prior = np.zeros(len(routes))
    names = sorted({r.destination for r in routes})
    total_weight = sum(weights.get(name, 0.0) for name in names)
    if total_weight <= 0:
        return prior
    for name in names:
        idx = [j for j, r in enumerate(routes) if r.destination == name]
        lengths = np.array([routes[j].length_m for j in idx])
        pull = np.exp(-beta * (lengths - lengths.min()) / max(lengths.min(), 1.0))
        share = weights.get(name, 0.0) / total_weight
        prior[idx] = crowd * share * pull / pull.sum()
    return prior


def fit_trips(
    passes: np.ndarray, targets: np.ndarray, prior: np.ndarray, prior_weight: float
) -> np.ndarray:
    """Non-negative walkers per route: sums near the targets, close to the prior."""
    pull = np.sqrt(prior_weight)
    a = np.vstack([passes, pull * np.eye(passes.shape[1])])
    b = np.concatenate([targets, pull * prior])
    trips, _ = nnls(a, b)
    return trips


def block_shares(profile: pd.DataFrame, block_minutes: int) -> np.ndarray:
    """Share of the hour in each block, or an even split when there is no profile."""
    n = 60 // block_minutes
    shares = np.zeros(n)
    for start, share in profile[["block_start", "share"]].itertuples(index=False):
        shares[int(start) // block_minutes] += share
    if shares.sum() <= 0:
        return np.full(n, 1.0 / n)
    return shares / shares.sum()


def crossings(
    trips: np.ndarray,
    passes: np.ndarray,
    offset: np.ndarray,
    hour_start: pd.Timestamp,
    shares: np.ndarray,
    block_minutes: int,
    speed: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One row per walker and sensor passed, with the time it passes.

    A walker's time at the first sensor on its route is drawn from the block shares,
    and it reaches later sensors at `speed` metres per second.
    """
    rows = []
    for j, n in enumerate(np.round(trips).astype(int)):
        on_route = np.flatnonzero(passes[:, j] > 0)
        if n == 0 or len(on_route) == 0:
            continue
        first = on_route[np.argmin(offset[on_route, j])]
        blocks = rng.choice(len(shares), size=n, p=shares)
        minutes = blocks * block_minutes + rng.random(n) * block_minutes
        for i in on_route:
            lag = (offset[i, j] - offset[first, j]) / speed / 60
            times = hour_start + pd.to_timedelta(minutes + lag, unit="min")
            rows.append(pd.DataFrame({"sensor_index": i, "time": times}))
    if not rows:
        return pd.DataFrame(columns=["sensor_index", "time"])
    return pd.concat(rows, ignore_index=True)


def to_blocks(
    passed: pd.DataFrame, sensors: list[int], block_minutes: int
) -> pd.DataFrame:
    """Simulated counts per sensor and block."""
    if passed.empty:
        return pd.DataFrame(columns=["location_id", "block", "total"])
    out = passed.assign(
        location_id=np.asarray(sensors)[passed["sensor_index"].to_numpy(int)],
        block=pd.to_datetime(passed["time"]).dt.floor(f"{block_minutes}min"),
    )
    counts = out.groupby(["location_id", "block"], as_index=False).size()
    return counts.rename(columns={"size": "total"})


def compare(
    simulated: pd.DataFrame, observed_series: pd.DataFrame, fitted: list[int]
) -> pd.DataFrame:
    """Per sensor: totals, peak blocks and counts, and the correlation of the series."""
    rows = []
    sensors = set(observed_series["location_id"]) | set(simulated["location_id"])
    for sensor in sorted(sensors):
        obs = observed_series[observed_series["location_id"] == sensor]
        sim = simulated[simulated["location_id"] == sensor]
        both = pd.concat(
            [
                obs.set_index("block")["total"].rename("obs"),
                sim.set_index("block")["total"].rename("sim"),
            ],
            axis=1,
            sort=True,
        ).fillna(0)
        row = {
            "location_id": sensor,
            "used_in_fit": sensor in fitted,
            "observed_total": int(both["obs"].sum()),
            "simulated_total": int(both["sim"].sum()),
        }
        for name in ("obs", "sim"):
            if both[name].sum() > 0:
                row[f"{name}_peak_block"] = both[name].idxmax()
                row[f"{name}_peak_count"] = int(both[name].max())
        if both["obs"].std() > 0 and both["sim"].std() > 0:
            row["correlation"] = round(float(both["obs"].corr(both["sim"])), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def load_area_graph(settings, area, sensor_segments):
    """The routing graph around the area's sensors, and each node's coordinates."""
    import geopandas as gpd

    cfg = area["walkers"]
    edges = gpd.read_parquet(REPO_ROOT / settings["routing_edges"])
    edges = edges[edges["in_main_component"].fillna(False).astype(bool)]
    near = edges[edges["segment_id"].isin(list(sensor_segments.values()))]
    x0, y0, x1, y1 = near.total_bounds
    pad = cfg["area_margin_m"]
    edges = edges.cx[x0 - pad : x1 + pad, y0 - pad : y1 + pad]
    ends = {}
    for u, v, geom in edges[["u", "v", "geometry"]].itertuples(index=False):
        coords = list(geom.coords)
        ends.setdefault(u, coords[0])
        ends.setdefault(v, coords[-1])
    return edges, build_graph(edges), ends


def origin_nodes(settings, area, edges) -> list[Any]:
    """Nodes on the segments within origin_radius_m of the event venue."""
    venue = pd.read_parquet(REPO_ROOT / settings["venue_dist"])
    near = venue[
        (venue["venue"] == area["event_venue"])
        & (venue["dist_m"] <= area["walkers"]["origin_radius_m"])
    ]
    gates = edges[edges["segment_id"].isin(near["segment_id"])]
    origins = sorted(set(gates["u"]) | set(gates["v"]))
    if not origins:
        names = sorted(venue["venue"].unique())
        raise SystemExit(
            f"no segments within origin_radius_m of {area['event_venue']}; "
            f"venues in the file: {names}"
        )
    return origins


def destination_nodes(area, crs, ends) -> dict[str, Any]:
    """The graph node nearest each destination's latitude and longitude."""
    from pyproj import Transformer

    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ids = np.array(list(ends))
    xy = np.array([ends[n] for n in ids])
    found = {}
    for name, spec in area["walkers"]["destinations"].items():
        x, y = to_grid.transform(spec["lon"], spec["lat"])
        found[name] = ids[np.argmin(((xy - (x, y)) ** 2).sum(axis=1))]
    return found


def run(
    area_name: str, night: str, settings: dict[str, Any] | None = None
) -> pd.DataFrame:
    settings = settings or observed.load_settings()
    area = settings["areas"][area_name]
    cfg = area["walkers"]
    rng = np.random.default_rng(cfg["seed"])

    seg_map = pd.read_parquet(REPO_ROOT / settings["sensor_segment_map"])
    segment_of = dict(zip(seg_map["location_id"], seg_map["segment_id"]))
    sensors = [s for s in area["sensors"] if s in segment_of]
    sensor_segments = {s: segment_of[s] for s in sensors}

    edges, graph, ends = load_area_graph(settings, area, sensor_segments)
    origins = origin_nodes(settings, area, edges)
    destinations = destination_nodes(area, edges.crs, ends)
    weights = {name: d["weight"] for name, d in cfg["destinations"].items()}
    routes = sample_routes(
        graph,
        origins,
        destinations,
        cfg["route_draws"],
        cfg["route_noise"],
        cfg["seed"],
    )
    passes, offset = incidence(routes, sensor_segments)
    log.info(
        "%d origins, %d routes; routes past each sensor: %s",
        len(origins),
        len(routes),
        dict(zip(sensors, passes.sum(axis=1).astype(int).tolist())),
    )

    hourly = pd.read_parquet(
        REPO_ROOT / settings["counts_hourly"],
        columns=["location_id", "ts_local", "pedestriancount", "direction_1"],
    )
    hourly = hourly[hourly["location_id"].isin(sensors)]
    hourly["ts_local"] = pd.to_datetime(hourly["ts_local"]).dt.tz_convert(TZ)
    profiles = pd.read_parquet(REPO_ROOT / settings["minute_profiles"])
    profiles = profiles[
        (profiles["location_id"] == area["lead_sensor"])
        & (profiles["hour_type"] == "event_post")
    ]

    fitted = [s for s in sensors if s in cfg["fit_sensors"]]
    fit_rows = [sensors.index(s) for s in fitted]
    day = pd.Timestamp(night, tz=TZ)
    day_type = "weekend" if day.day_of_week >= 5 else "weekday"
    parts, direction_share = [], {}
    for hour in cfg["hours"]:
        start = day + pd.Timedelta(hours=hour)
        now = hourly[hourly["ts_local"] == start].set_index("location_id")
        counts = now["pedestriancount"]
        targets = np.array([counts.get(s, 0.0) for s in fitted], dtype=float)
        crowd = float(targets.max(initial=0.0))
        prior = route_prior(routes, weights, crowd, cfg["length_beta"])
        trips = fit_trips(passes[fit_rows], targets, prior, cfg["prior_weight"])
        profile = profiles[
            (profiles["day_type"] == day_type) & (profiles["hour_local"] == hour % 24)
        ]
        shares = block_shares(profile, area["block_minutes"])
        parts.append(
            crossings(
                trips,
                passes,
                offset,
                start,
                shares,
                area["block_minutes"],
                cfg["speed_mps"],
                rng,
            )
        )
        for s in sensors:
            if counts.get(s, 0) > 0:
                share_1 = now.loc[s, "direction_1"] / counts[s]
                direction_share[(s, start.hour)] = share_1
        log.info(
            "hour %02d: fitted targets %s, walkers %d",
            hour,
            targets.astype(int).tolist(),
            round(trips.sum()),
        )

    simulated = to_blocks(
        pd.concat(parts, ignore_index=True), sensors, area["block_minutes"]
    )
    share = [
        direction_share.get((s, b.hour), 0.5)
        for s, b in simulated[["location_id", "block"]].itertuples(index=False)
    ]
    simulated["direction_1"] = simulated["total"] * np.asarray(share, dtype=float)
    simulated["direction_2"] = simulated["total"] - simulated["direction_1"]

    out = REPO_ROOT / settings["output_dir"] / area_name
    series = pd.read_parquet(out / "observed_egress.parquet")
    series = series[series["night"].astype(str) == night]
    if series.empty:
        raise SystemExit(f"no observed series for {night}: run observed.py first")
    in_window = simulated["block"].between(series["block"].min(), series["block"].max())
    simulated = simulated[in_window]
    result = compare(simulated, series, fitted)
    simulated.to_parquet(out / f"walkers_{night}.parquet", index=False)
    result.to_csv(out / f"walkers_compare_{night}.csv", index=False)
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description="Route-sampling walkers for event egress")
    ap.add_argument("--area", default="melbourne_park")
    ap.add_argument("--night", required=True, help="date of the event, YYYY-MM-DD")
    args = ap.parse_args()
    result = run(args.area, args.night)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
