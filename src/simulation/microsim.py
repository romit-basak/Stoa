"""JuPedSim on a corridor around the lead sensor (getting_started.md, step 3).

The walkers decide who leaves the venue, by which route and when they pass the lead
sensor. This moves the walkers on routes past the lead sensor through a corridor
around it, second by second, with the Collision Free Speed Model, so people slow down
where it gets crowded. A measurement line across the path at the sensor counts them
per 5-minute block, to compare with the observed counts and with the walkers, who all
walk at one fixed speed.

The corridor is each route's path from `before_m` before the sensor to `after_m` after
it, widened by `half_width_m` on each side and merged. Each walker enters at its
route's start of the corridor at the time that gets it to the sensor when the walkers
said, follows waypoints along its route, and leaves at its route's end of the corridor.

JuPedSim (LGPL-3.0) runs in its own environment, outside the main project
(getting_started.md), so run this with that environment's Python, from the repository:

    jps-env/bin/python -m src.simulation.microsim --area melbourne_park --night 2026-10-27

Run walkers.py for the night first. Results are planning guidance for outdoor approach
areas, never a crowd-safety assessment (core rule 7).
"""

from __future__ import annotations

import argparse
import logging
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from src.simulation import observed

REPO_ROOT = observed.REPO_ROOT
TZ = observed.TZ

log = logging.getLogger("stoa.simulation.microsim")


def chain(lines: list[np.ndarray]) -> np.ndarray:
    """One polyline from consecutive segments, each turned to start where the last ends."""
    if not lines:
        return np.empty((0, 2))
    if len(lines) == 1:
        return np.asarray(lines[0], dtype=float)
    first, second = np.asarray(lines[0], float), np.asarray(lines[1], float)
    ends = (second[0], second[-1])
    if min(np.linalg.norm(first[0] - e) for e in ends) < min(
        np.linalg.norm(first[-1] - e) for e in ends
    ):
        first = first[::-1]
    points = [first]
    for line in lines[1:]:
        line = np.asarray(line, dtype=float)
        tail = points[-1][-1]
        if np.linalg.norm(line[-1] - tail) < np.linalg.norm(line[0] - tail):
            line = line[::-1]
        points.append(line[1:])
    return np.vstack(points)


def distances(line: np.ndarray) -> np.ndarray:
    """Distance along the polyline at each of its points."""
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def point_at(line: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """The point `d` metres along the polyline, and the unit direction there."""
    along = distances(line)
    d = float(np.clip(d, 0.0, along[-1]))
    i = int(np.clip(np.searchsorted(along, d, side="right") - 1, 0, len(line) - 2))
    step = line[i + 1] - line[i]
    length = np.linalg.norm(step)
    direction = step / length if length > 0 else np.array([1.0, 0.0])
    share = (d - along[i]) / length if length > 0 else 0.0
    return line[i] + share * step, direction


def cut(line: np.ndarray, start: float, end: float) -> np.ndarray:
    """The part of the polyline between `start` and `end` metres along it."""
    along = distances(line)
    start, end = max(0.0, start), min(float(along[-1]), end)
    inside = line[(along > start) & (along < end)]
    first, _ = point_at(line, start)
    last, _ = point_at(line, end)
    return np.vstack([first, inside, last])


def project(line: np.ndarray, point: np.ndarray) -> float:
    """Distance along the polyline to the point on it nearest `point`."""
    along = distances(line)
    best, best_d = np.inf, 0.0
    for i in range(len(line) - 1):
        a, b = line[i], line[i + 1]
        step = b - a
        length2 = float(step @ step)
        t = 0.0 if length2 == 0 else float(np.clip((point - a) @ step / length2, 0, 1))
        gap = np.linalg.norm(a + t * step - point)
        if gap < best:
            best, best_d = gap, along[i] + t * np.sqrt(length2)
    return float(best_d)


def waypoints(window: np.ndarray, spacing: float) -> list[np.ndarray]:
    """Points every `spacing` metres along the window, short of its end."""
    length = distances(window)[-1]
    return [
        point_at(window, d)[0] for d in np.arange(spacing, length - spacing, spacing)
    ]


def crossed(before: float, after: float) -> bool:
    """True when a signed distance along the path goes from behind the line to past it."""
    return before < 0 <= after


def to_blocks(times: pd.Series, block_minutes: int) -> pd.Series:
    """Counts per block of `block_minutes`, indexed by the block's start."""
    if len(times) == 0:
        return pd.Series(dtype=float)
    blocks = pd.to_datetime(times).dt.floor(f"{block_minutes}min")
    return blocks.value_counts().sort_index().astype(float)


def summarise(series: dict[str, pd.Series], reference: str) -> pd.DataFrame:
    """Per source: total, peak block, peak count and correlation with the reference."""
    table = pd.concat(series, axis=1, sort=True).fillna(0.0)
    rows = []
    for name in table.columns:
        row = {"source": name, "total": int(table[name].sum())}
        if table[name].sum() > 0:
            row["peak_block"] = table[name].idxmax()
            row["peak_count"] = int(table[name].max())
        if name != reference and table[name].std() > 0 and table[reference].std() > 0:
            row["correlation"] = round(float(table[name].corr(table[reference])), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def agent_parameters(jps: Any, **values: Any) -> Any:
    """Collision Free Speed Model agent parameters, for either name of the speed field."""
    try:
        return jps.CollisionFreeSpeedModelAgentParameters(**values)
    except TypeError:
        values["v0"] = values.pop("desired_speed")
        return jps.CollisionFreeSpeedModelAgentParameters(**values)


def simulate(
    jps: Any,
    corridor: Any,
    windows: dict[int, np.ndarray],
    entries: pd.DataFrame,
    line_point: np.ndarray,
    line_direction: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[list[float], int]:
    """Run JuPedSim; seconds at which agents cross the line, and the most at once."""
    from shapely.geometry import Point

    sim = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(), geometry=corridor, dt=cfg["dt"]
    )
    journeys = {}
    for route, window in windows.items():
        stages = [
            sim.add_waypoint_stage(tuple(p), cfg["waypoint_reach_m"])
            for p in waypoints(window, cfg["waypoint_spacing_m"])
        ]
        exit_area = (
            Point(window[-1]).buffer(cfg["exit_radius_m"]).intersection(corridor)
        )
        stages.append(sim.add_exit_stage(exit_area))
        journey = jps.JourneyDescription(stages)
        for here, there in pairwise(stages):
            journey.set_transition_for_stage(
                here, jps.Transition.create_fixed_transition(there)
            )
        journeys[route] = (sim.add_journey(journey), stages[0])

    rng = np.random.default_rng(cfg["seed"])
    offsets = cfg["lateral_offsets_m"]
    queue = entries.sort_values("entry_s").to_dict("records")
    upcoming, retry = 0, []
    sides, crossings, most = {}, [], 0
    steps = max(1, round(cfg["check_s"] / cfg["dt"]))
    limit = cfg["max_minutes"] * 60
    reach = cfg["half_width_m"] + 1.0
    across = np.array([-line_direction[1], line_direction[0]])

    def busy() -> bool:
        return upcoming < len(queue) or bool(retry) or sim.agent_count() > 0

    while busy() and sim.elapsed_time() < limit:
        now = sim.elapsed_time()
        due, retry = retry, []
        while upcoming < len(queue) and queue[upcoming]["entry_s"] <= now:
            due.append(queue[upcoming])
            upcoming += 1
        for agent in due:
            window = windows[agent["route"]]
            start, direction = point_at(window, cfg["entry_inset_m"])
            normal = np.array([-direction[1], direction[0]])
            spot = start + normal * offsets[rng.integers(len(offsets))]
            if not corridor.contains(Point(spot)):
                spot = start
            journey, stage = journeys[agent["route"]]
            try:
                sim.add_agent(
                    agent_parameters(
                        jps,
                        journey_id=journey,
                        stage_id=stage,
                        position=tuple(spot),
                        desired_speed=cfg["speed_mps"],
                        radius=cfg["agent_radius_m"],
                    )
                )
            except RuntimeError:
                retry.append(agent)
        sim.iterate(steps)
        most = max(most, sim.agent_count())
        for walker in sim.agents():
            offset = np.asarray(walker.position) - line_point
            along = float(offset @ line_direction)
            last = sides.get(walker.id)
            near = abs(float(offset @ across)) <= reach
            if last is not None and near and crossed(last, along):
                crossings.append(sim.elapsed_time())
            sides[walker.id] = along
    if busy():
        log.warning(
            "stopped at %d minutes with %d waiting and %d walking",
            cfg["max_minutes"],
            len(queue) - upcoming + len(retry),
            sim.agent_count(),
        )
    return crossings, most


def run(area_name: str, night: str) -> pd.DataFrame:
    import geopandas as gpd
    import jupedsim as jps
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    settings = observed.load_settings()
    area = settings["areas"][area_name]
    cfg = dict(area["microsim"])
    cfg["speed_mps"] = area["walkers"]["speed_mps"]
    lead = area["lead_sensor"]
    out = REPO_ROOT / settings["output_dir"] / area_name

    routes = pd.read_parquet(out / f"walkers_routes_{night}.parquet")
    agents = pd.read_parquet(out / f"walkers_agents_{night}.parquet")
    seg_map = pd.read_parquet(REPO_ROOT / settings["sensor_segment_map"])
    lead_segment = dict(zip(seg_map["location_id"], seg_map["segment_id"]))[lead]
    used = routes[routes["route"].isin(agents["route"].unique())]
    needed = {s for segs in used["segments"] for s in segs} | {lead_segment}
    edges = gpd.read_parquet(REPO_ROOT / settings["routing_edges"])
    edges = edges[edges["segment_id"].isin(list(needed))]
    geometry_of = dict(zip(edges["segment_id"], edges.geometry))
    centre = geometry_of[lead_segment].interpolate(0.5, normalized=True)
    line_point = np.array([centre.x, centre.y])

    windows, shift = {}, {}
    for route, segs in zip(used["route"], used["segments"]):
        lines = [np.asarray(geometry_of[s].coords) for s in segs if s in geometry_of]
        path = chain(lines)
        at = project(path, line_point)
        start = at - cfg["before_m"]
        windows[int(route)] = cut(path, start, at + cfg["after_m"])
        shift[int(route)] = at - max(0.0, start)
    first = next(iter(windows))
    _, line_direction = point_at(windows[first], shift[first])

    pieces = [
        LineString(w).buffer(cfg["half_width_m"], cap_style="flat")
        for w in windows.values()
    ]
    corridor = unary_union(pieces).buffer(0)
    if corridor.geom_type == "MultiPolygon":
        corridor = next(g for g in corridor.geoms if g.contains(Point(line_point)))
    corridor = corridor.simplify(0.2)

    times = pd.to_datetime(agents["time"])
    walk = agents["route"].map(shift) / cfg["speed_mps"]
    entry = times - pd.to_timedelta(walk, unit="s")
    zero = entry.min()
    entries = pd.DataFrame(
        {
            "route": agents["route"].astype(int),
            "entry_s": (entry - zero).dt.total_seconds(),
        }
    )
    log.info(
        "%d walkers on %d routes past sensor %d; corridor %.0f m2",
        len(entries),
        len(windows),
        lead,
        corridor.area,
    )

    seconds, most = simulate(
        jps, corridor, windows, entries, line_point, line_direction, cfg
    )
    crossing_times = pd.Series(zero + pd.to_timedelta(seconds, unit="s"))
    egress = pd.read_parquet(out / "observed_egress.parquet")
    egress = egress[
        (egress["night"].astype(str) == night) & (egress["location_id"] == lead)
    ]
    block = area["block_minutes"]
    series = {
        "observed": egress.set_index("block")["total"].astype(float),
        "walkers": to_blocks(times, block),
        "jupedsim": to_blocks(crossing_times, block),
    }
    window = (egress["block"].min(), egress["block"].max())
    series = {
        k: v[(v.index >= window[0]) & (v.index <= window[1])] for k, v in series.items()
    }
    result = summarise(series, "observed")
    result["most_agents_at_once"] = np.where(
        result["source"] == "jupedsim", most, np.nan
    )
    pd.concat(series, axis=1, sort=True).fillna(0.0).to_parquet(
        out / f"microsim_{night}.parquet"
    )
    result.to_csv(out / f"microsim_compare_{night}.csv", index=False)
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description="JuPedSim around the lead sensor")
    ap.add_argument("--area", default="melbourne_park")
    ap.add_argument("--night", required=True, help="date of the event, YYYY-MM-DD")
    args = ap.parse_args()
    result = run(args.area, args.night)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
