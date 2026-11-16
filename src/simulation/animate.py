"""An animated GIF of the walkers leaving an event, for the docs and README.

Each dot is a walker moving along its route at walking speed, from its start time
(walkers.py), over the streets of the routing graph in light grey. Each sensor shows
the observed count in the current 5-minute block (observed.py), so the simulation can
be compared with reality, and a clock shows Melbourne time.

    .venv/bin/python -m src.simulation.animate --area melbourne_park --night 2026-10-27

Run observed.py and walkers.py for the night first. Settings are under `animation:` in
configs/simulation.yaml. Planning illustration only, never a crowd-safety forecast
(core rule 7).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.simulation import observed
from src.simulation.figures import FIGURES, lines_of, pyplot
from src.simulation.microsim import chain, distances

REPO_ROOT = observed.REPO_ROOT
TZ = observed.TZ
CAPTION = (
    "Simulated walkers fitted to observed hourly counts at four sensors, {date} "
    "(1 in {sample} shown).\n"
    "Planning illustration, outdoor streets only; not a crowd-safety forecast."
)

log = logging.getLogger("stoa.simulation.animate")


def positions(
    along: np.ndarray,
    line: np.ndarray,
    starts_s: np.ndarray,
    now_s: float,
    speed: float,
) -> np.ndarray:
    """Where each walker on one route is at `now_s`: x, y, or NaN off the route."""
    walked = speed * (now_s - starts_s)
    on_route = (walked >= 0) & (walked <= along[-1])
    xy = np.full((len(starts_s), 2), np.nan)
    xy[on_route, 0] = np.interp(walked[on_route], along, line[:, 0])
    xy[on_route, 1] = np.interp(walked[on_route], along, line[:, 1])
    return xy


def block_counts(series: pd.DataFrame, block_minutes: int) -> dict:
    """Observed count per (sensor, block start), from the observed egress series."""
    return {
        (int(s), b): int(c)
        for s, b, c in series[["location_id", "block", "total"]].itertuples(index=False)
    }


def observed_now(counts: dict, sensor: int, now: pd.Timestamp, minutes: int) -> int:
    """The observed count at a sensor in the block that contains `now`."""
    return counts.get((sensor, now.floor(f"{minutes}min")), 0)


def render(
    streets: list[np.ndarray],
    routes: dict[int, tuple[np.ndarray, np.ndarray]],
    walkers: pd.DataFrame,
    sensors: list[dict[str, Any]],
    counts: dict,
    frames: list[pd.Timestamp],
    cfg: dict[str, Any],
    caption: str,
    path: Path,
    route_walkers: dict[int, float] | None = None,
    places: list[dict[str, Any]] | None = None,
) -> Path:
    """Draw the animation and save it as a GIF.

    With `route_walkers`, each route is drawn faintly under the dots, darker and wider
    the more walkers take it, as in the still route map. `places` are named landmarks
    (the venue and the stations) marked with their names.
    """
    plt = pyplot()
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    points = [line for line, _ in routes.values()] + [
        np.array([[s["x"], s["y"]]]) for s in sensors
    ]
    xy = np.vstack(points)
    pad = cfg["margin_m"]
    x0, x1 = xy[:, 0].min() - pad, xy[:, 0].max() + pad
    y0, y1 = xy[:, 1].min() - pad, xy[:, 1].max() + pad
    width_in = cfg["width_px"] / cfg["dpi"]
    map_in = width_in * float(np.clip((y1 - y0) / (x1 - x0), 0.45, 1.2))
    fig, ax = plt.subplots(figsize=(width_in, map_in + 0.6), dpi=cfg["dpi"])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.6 / (map_in + 0.6))
    ax.add_collection(
        LineCollection(streets, colors="#d4d4d4", linewidths=0.5), autolim=False
    )
    if route_walkers:
        cmap = plt.get_cmap("plasma_r")
        top = max(route_walkers.values()) or 1.0
        for route, (line, _) in routes.items():
            share = route_walkers.get(route, 0.0) / top
            if share <= 0:
                continue
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=cmap(0.15 + 0.85 * share),
                linewidth=0.8 + 3.0 * share,
                alpha=0.45,
                solid_capstyle="round",
                zorder=2,
            )
    ax.update_datalim([[x0, y0], [x1, y1]])
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.set_axis_off()

    for place in places or []:
        ax.scatter(
            place["x"],
            place["y"],
            marker=place["marker"],
            s=place["size"],
            color=place["colour"],
            zorder=4,
        )
        right_side = place["x"] > (x0 + x1) / 2
        ax.annotate(
            place["name"],
            (place["x"], place["y"]),
            xytext=(-7, -10) if right_side else (7, -10),
            ha="right" if right_side else "left",
            textcoords="offset points",
            fontsize=8,
            color=place["colour"],
            bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.7},
            zorder=5,
        )
    dots = ax.scatter([], [], s=6, color="#e8710a", alpha=0.8, zorder=3)
    ax.scatter(
        [s["x"] for s in sensors],
        [s["y"] for s in sensors],
        s=60,
        color="#d62728",
        edgecolor="white",
        zorder=4,
    )
    labels = [
        ax.annotate(
            "",
            (s["x"], s["y"]),
            xytext=s.get("label_offset", (6, 6)),
            textcoords="offset points",
            fontsize=8,
            color="#d62728",
            fontweight="bold",
            zorder=5,
        )
        for s in sensors
    ]
    box = {"boxstyle": "round", "fc": "white", "ec": "#bbbbbb", "alpha": 0.9}
    legend = "Orange dots: simulated walkers\n"
    if route_walkers:
        legend += "Lines: their routes, darker where more walk\n"
    legend += "Red: sensors, observed people per 5 minutes"
    clock = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=13,
        va="top",
        fontweight="bold",
        bbox=box,
        zorder=6,
    )
    ax.text(
        0.98,
        0.97,
        legend,
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        bbox=box,
        zorder=6,
    )
    ax.text(
        0.99,
        0.01,
        "\u00a9 OpenStreetMap contributors",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="bottom",
        color="#555555",
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7.5)

    groups = {
        route: walkers.loc[walkers["route"] == route, "start_s"].to_numpy()
        for route in routes
    }
    zero = frames[0]

    def draw(i: int):
        now = frames[i]
        now_s = (now - zero).total_seconds()
        found = [
            positions(along, line, groups[route], now_s, cfg["speed_mps"])
            for route, (line, along) in routes.items()
        ]
        here = np.vstack(found) if found else np.empty((0, 2))
        here = here[~np.isnan(here[:, 0])]
        dots.set_offsets(here if len(here) else np.empty((0, 2)))
        for label, sensor in zip(labels, sensors):
            count = observed_now(counts, sensor["id"], now, cfg["block_minutes"])
            label.set_text(f"{sensor['id']}: {count}")
        clock.set_text(now.strftime("%H:%M"))
        return [dots, clock, *labels]

    animation = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=cfg["fps"]), dpi=cfg["dpi"])
    plt.close(fig)
    return path


def landmarks(settings, area, edges, geometry_of) -> list[dict[str, Any]]:
    """The venue and the walkers' destinations, with names, in the map's coordinates."""
    from pyproj import Transformer

    places = []
    venue = pd.read_parquet(REPO_ROOT / settings["venue_dist"])
    near = venue[
        (venue["venue"] == area["event_venue"])
        & (venue["dist_m"] <= area["walkers"]["origin_radius_m"])
    ]
    gates = [geometry_of[s].centroid for s in near["segment_id"] if s in geometry_of]
    if gates:
        places.append(
            {
                "name": area["animation"]["venue_name"],
                "x": float(np.mean([g.x for g in gates])),
                "y": float(np.mean([g.y for g in gates])),
                "marker": "*",
                "size": 220,
                "colour": "black",
            }
        )
    to_grid = Transformer.from_crs("EPSG:4326", edges.crs, always_xy=True)
    for name, spec in area["walkers"]["destinations"].items():
        x, y = to_grid.transform(spec["lon"], spec["lat"])
        places.append(
            {
                "name": name.replace("_", " ").title(),
                "x": x,
                "y": y,
                "marker": "^",
                "size": 70,
                "colour": "#1f6fb4",
            }
        )
    return places


def run(area_name: str, night: str) -> Path:
    import geopandas as gpd

    settings = observed.load_settings()
    area = settings["areas"][area_name]
    cfg = dict(area["animation"])
    cfg["speed_mps"] = area["walkers"]["speed_mps"]
    cfg["block_minutes"] = area["block_minutes"]
    out = REPO_ROOT / settings["output_dir"] / area_name

    routes = pd.read_parquet(out / f"walkers_routes_{night}.parquet")
    starts = pd.read_parquet(out / f"walkers_starts_{night}.parquet")
    rng = np.random.default_rng(area["walkers"]["seed"])
    keep = rng.random(len(starts)) < 1.0 / cfg["sample_every"]
    starts = starts[keep].reset_index(drop=True)

    seg_map = pd.read_parquet(REPO_ROOT / settings["sensor_segment_map"])
    segment_of = dict(zip(seg_map["location_id"], seg_map["segment_id"]))
    edges = gpd.read_parquet(REPO_ROOT / settings["routing_edges"])
    used = routes[routes["walkers"] >= 1]
    needed = {s for segs in used["segments"] for s in segs}
    needed |= {segment_of[s] for s in area["sensors"] if s in segment_of}
    x0, y0, x1, y1 = edges[edges["segment_id"].isin(list(needed))].total_bounds
    pad = cfg["margin_m"]
    nearby = edges.cx[x0 - pad : x1 + pad, y0 - pad : y1 + pad]
    geometry_of = dict(zip(nearby["segment_id"], nearby.geometry))
    streets = [line for g in nearby.geometry for line in lines_of(g)]

    paths = {}
    for route, segs in zip(used["route"], used["segments"]):
        parts = [np.asarray(geometry_of[s].coords) for s in segs if s in geometry_of]
        if len(parts) < len(segs):
            continue
        line = chain(parts)
        paths[int(route)] = (line, distances(line))

    day = pd.Timestamp(night)
    first = (day + pd.Timedelta(cfg["start"] + ":00")).tz_localize(TZ)
    last = (day + pd.Timedelta(cfg["end"] + ":00")).tz_localize(TZ)
    frames = list(pd.date_range(first, last, freq=f"{cfg['frame_minutes']}min"))
    begin = pd.to_datetime(starts["start"]).dt.tz_convert(TZ)
    starts = starts.assign(start_s=(begin - first).dt.total_seconds())
    starts = starts[starts["route"].isin(list(paths))]

    egress = pd.read_parquet(out / "observed_egress.parquet")
    egress = egress[egress["night"].astype(str) == night]
    counts = block_counts(egress, area["block_minutes"])
    offsets = cfg.get("label_offsets", {})
    sensors = []
    for sensor in area["sensors"]:
        segment = segment_of.get(sensor)
        if segment not in geometry_of:
            continue
        point = geometry_of[segment].interpolate(0.5, normalized=True)
        sensors.append(
            {
                "id": sensor,
                "x": point.x,
                "y": point.y,
                "label_offset": tuple(offsets.get(sensor, (6, 6))),
            }
        )

    date = day.strftime("%d %b %Y").lstrip("0")
    caption = CAPTION.format(date=date, sample=cfg["sample_every"])
    if cfg["sample_every"] == 1:
        caption = caption.replace(" (1 in 1 shown)", "")
    places = landmarks(settings, area, edges, geometry_of)
    path = FIGURES / f"{area_name}_egress_{night}.gif"
    log.info(
        "%d walkers drawn (1 in %d), %d routes, %d frames",
        len(starts),
        cfg["sample_every"],
        len(paths),
        len(frames),
    )
    route_walkers = None
    if cfg.get("draw_routes", False):
        route_walkers = {
            int(r): float(w)
            for r, w in zip(used["route"], used["walkers"])
            if int(r) in paths
        }
    return render(
        streets,
        paths,
        starts,
        sensors,
        counts,
        frames,
        cfg,
        caption,
        path,
        route_walkers,
        places,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description="Animated GIF of the walkers")
    ap.add_argument("--area", default="melbourne_park")
    ap.add_argument("--night", required=True, help="date of the event, YYYY-MM-DD")
    args = ap.parse_args()
    path = run(args.area, args.night)
    size = path.stat().st_size / 1e6
    log.info("wrote %s (%.1f MB)", path, size)


if __name__ == "__main__":
    main()
