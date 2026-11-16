"""Figures of the observed egress and the walkers, for the docs.

    .venv/bin/python -m src.simulation.figures --area melbourne_park \\
        --map-night 2026-10-27 --nights 2026-10-26 2026-10-27 2026-10-28 2026-10-29

Writes PNG files to docs/simulation/figures/:

- walkers_map_<night>.png: the streets, the walkers' routes coloured and sized by how
  many walkers take them, the venue, the destinations, and each sensor with its observed
  and simulated totals;
- sensor_<id>_nights.png: observed and simulated counts per 5 minutes at the lead
  sensor, one panel per night;
- observed_<night>.png: observed counts per 5 minutes at every sensor on one night.

Run observed.py and walkers.py for the nights first.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.simulation import observed

REPO_ROOT = observed.REPO_ROOT
TZ = observed.TZ
FIGURES = REPO_ROOT / "docs" / "simulation" / "figures"
START_HOUR = 22

log = logging.getLogger("stoa.simulation.figures")


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def clock(minutes: float, _pos: Any = None) -> str:
    """Minutes after 22:00 as a clock time."""
    total = round(START_HOUR * 60 + minutes) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def minutes_after_start(blocks: pd.Series, night: str) -> np.ndarray:
    start = pd.Timestamp(night, tz=TZ) + pd.Timedelta(hours=START_HOUR)
    return ((pd.to_datetime(blocks) - start).dt.total_seconds() / 60).to_numpy()


def walkers_map(
    streets: list[np.ndarray],
    routes: list[tuple[list[np.ndarray], float]],
    sensors: list[dict[str, Any]],
    venue: tuple[float, float] | None,
    destinations: dict[str, tuple[float, float]],
    title: str,
    path: Path,
    margin_m: float = 150.0,
) -> Path:
    """Streets, routes sized and coloured by walkers, the venue and the sensors."""
    plt = pyplot()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(10, 9))
    for line in streets:
        ax.plot(line[:, 0], line[:, 1], color="#cfcfcf", linewidth=0.4, zorder=1)

    used = sorted((r for r in routes if r[1] >= 1), key=lambda r: r[1])
    top = max((w for _, w in used), default=1.0)
    norm = Normalize(vmin=0, vmax=top)
    cmap = plt.get_cmap("plasma_r")
    for parts, walkers in used:
        for line in parts:
            ax.plot(
                line[:, 0],
                line[:, 1],
                color=cmap(norm(walkers)),
                linewidth=0.8 + 5.0 * walkers / top,
                solid_capstyle="round",
                zorder=2,
            )
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
    bar.set_label("walkers on the route")

    points = [line for parts, _ in used for line in parts]
    if venue is not None:
        ax.scatter(*venue, marker="*", s=380, color="black", zorder=4)
        ax.annotate(
            "venue", venue, xytext=(8, -14), textcoords="offset points", fontsize=9
        )
        points.append(np.array([venue]))
    for name, (x, y) in destinations.items():
        ax.scatter(x, y, marker="^", s=110, color="#1f6fb4", zorder=4)
        ax.annotate(
            name.replace("_", " "),
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#1f6fb4",
        )
        points.append(np.array([[x, y]]))
    details = []
    for sensor in sensors:
        ax.scatter(
            sensor["x"], sensor["y"], s=90, color="#d62728", edgecolor="white", zorder=5
        )
        ax.annotate(
            sensor["label"],
            (sensor["x"], sensor["y"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#d62728",
            zorder=6,
        )
        if sensor.get("detail"):
            details.append(sensor["detail"])
        points.append(np.array([[sensor["x"], sensor["y"]]]))
    if details:
        ax.text(
            0.01,
            0.01,
            "\n".join(["Sensor: observed / simulated", *details]),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"boxstyle": "round", "fc": "white", "ec": "#d62728", "alpha": 0.9},
            zorder=7,
        )

    if points:
        xy = np.vstack(points)
        ax.set_xlim(xy[:, 0].min() - margin_m, xy[:, 0].max() + margin_m)
        ax.set_ylim(xy[:, 1].min() - margin_m, xy[:, 1].max() + margin_m)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def nights_figure(panels: list[dict[str, Any]], sensor: int, path: Path) -> Path:
    """Observed and simulated counts per 5 minutes at one sensor, a panel per night."""
    plt = pyplot()
    from matplotlib.ticker import FuncFormatter

    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3.4 * rows), sharey=True, squeeze=False
    )
    for ax, panel in zip(axes.ravel(), panels):
        ax.plot(
            panel["obs_minutes"],
            panel["obs_counts"],
            marker="o",
            markersize=3,
            color="black",
            label="observed",
        )
        ax.plot(
            panel["sim_minutes"],
            panel["sim_counts"],
            marker="o",
            markersize=3,
            color="#e8710a",
            label="walkers",
        )
        corr = panel.get("correlation")
        note = f", r = {corr:.2f}" if corr is not None and np.isfinite(corr) else ""
        ax.set_title(f"{panel['night']}{note}", fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(clock))
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("people per 5 minutes")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Sensor {sensor}: observed and simulated crowd leaving", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def observed_figure(
    series: dict[int, tuple[np.ndarray, np.ndarray]], night: str, path: Path
) -> Path:
    """Observed counts per 5 minutes at every sensor on one night."""
    plt = pyplot()
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for sensor, (minutes, counts) in sorted(series.items()):
        ax.plot(minutes, counts, marker="o", markersize=3, label=f"sensor {sensor}")
    ax.xaxis.set_major_formatter(FuncFormatter(clock))
    ax.set_ylabel("people per 5 minutes")
    ax.set_title(f"Observed crowd leaving after the concert, {night}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def lines_of(geometry) -> list[np.ndarray]:
    parts = getattr(geometry, "geoms", [geometry])
    return [np.asarray(part.coords) for part in parts]


def run(area_name: str, map_night: str, nights: list[str]) -> list[Path]:
    import geopandas as gpd
    from pyproj import Transformer

    settings = observed.load_settings()
    area = settings["areas"][area_name]
    out = REPO_ROOT / settings["output_dir"] / area_name
    lead = area["lead_sensor"]
    written = []

    routes = pd.read_parquet(out / f"walkers_routes_{map_night}.parquet")
    compare = pd.read_csv(out / f"walkers_compare_{map_night}.csv")
    seg_map = pd.read_parquet(REPO_ROOT / settings["sensor_segment_map"])
    segment_of = dict(zip(seg_map["location_id"], seg_map["segment_id"]))

    edges = gpd.read_parquet(REPO_ROOT / settings["routing_edges"])
    used = {s for segs in routes["segments"] for s in segs}
    used |= {segment_of[s] for s in area["sensors"] if s in segment_of}
    x0, y0, x1, y1 = edges[edges["segment_id"].isin(list(used))].total_bounds
    pad = 300
    nearby = edges.cx[x0 - pad : x1 + pad, y0 - pad : y1 + pad]
    geometry_of = dict(zip(nearby["segment_id"], nearby.geometry))
    streets = [line for g in nearby.geometry for line in lines_of(g)]
    route_lines = []
    for segs, walkers in zip(routes["segments"], routes["walkers"]):
        parts = [lines_of(geometry_of[s]) for s in segs if s in geometry_of]
        route_lines.append(([line for p in parts for line in p], float(walkers)))

    totals = compare.set_index("location_id")
    sensors = []
    for sensor in area["sensors"]:
        segment = segment_of.get(sensor)
        if segment not in geometry_of:
            continue
        point = geometry_of[segment].interpolate(0.5, normalized=True)
        detail = ""
        if sensor in totals.index:
            obs = int(totals.loc[sensor, "observed_total"])
            sim = int(totals.loc[sensor, "simulated_total"])
            detail = f"{sensor}: {obs:,} / {sim:,}"
        sensors.append(
            {"x": point.x, "y": point.y, "label": str(sensor), "detail": detail}
        )

    venue_dist = pd.read_parquet(REPO_ROOT / settings["venue_dist"])
    near = venue_dist[
        (venue_dist["venue"] == area["event_venue"])
        & (venue_dist["dist_m"] <= area["walkers"]["origin_radius_m"])
    ]
    gates = [geometry_of[s].centroid for s in near["segment_id"] if s in geometry_of]
    venue = (
        (float(np.mean([g.x for g in gates])), float(np.mean([g.y for g in gates])))
        if gates
        else None
    )
    to_grid = Transformer.from_crs("EPSG:4326", edges.crs, always_xy=True)
    destinations = {
        name: to_grid.transform(spec["lon"], spec["lat"])
        for name, spec in area["walkers"]["destinations"].items()
    }
    written.append(
        walkers_map(
            streets,
            route_lines,
            sensors,
            venue,
            destinations,
            f"Walkers leaving the concert, {map_night}",
            FIGURES / f"walkers_map_{map_night}.png",
        )
    )

    egress = pd.read_parquet(out / "observed_egress.parquet")
    egress["night"] = egress["night"].astype(str)
    panels = []
    for night in nights:
        obs = egress[(egress["night"] == night) & (egress["location_id"] == lead)]
        sim_path = out / f"walkers_{night}.parquet"
        if obs.empty or not sim_path.exists():
            log.warning("skipping %s: run observed.py and walkers.py first", night)
            continue
        sim = pd.read_parquet(sim_path)
        sim = sim[sim["location_id"] == lead]
        corr = pd.read_csv(out / f"walkers_compare_{night}.csv").set_index(
            "location_id"
        )
        panels.append(
            {
                "night": night,
                "obs_minutes": minutes_after_start(obs["block"], night),
                "obs_counts": obs["total"].to_numpy(),
                "sim_minutes": minutes_after_start(sim["block"], night),
                "sim_counts": sim["total"].to_numpy(),
                "correlation": corr["correlation"].get(lead, np.nan)
                if "correlation" in corr
                else np.nan,
            }
        )
    if panels:
        written.append(
            nights_figure(panels, lead, FIGURES / f"sensor_{lead}_nights.png")
        )

    one = egress[egress["night"] == map_night]
    series = {
        int(s): (minutes_after_start(g["block"], map_night), g["total"].to_numpy())
        for s, g in one.groupby("location_id")
    }
    if series:
        written.append(
            observed_figure(series, map_night, FIGURES / f"observed_{map_night}.png")
        )
    for path in written:
        log.info("wrote %s", path)
    return written


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description="Simulation figures for the docs")
    ap.add_argument("--area", default="melbourne_park")
    ap.add_argument("--map-night", required=True)
    ap.add_argument("--nights", nargs="+", required=True)
    args = ap.parse_args()
    run(args.area, args.map_night, args.nights)


if __name__ == "__main__":
    main()
