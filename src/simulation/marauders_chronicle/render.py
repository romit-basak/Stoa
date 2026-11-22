"""STOA: a vibrant, full-bleed Marauder's Map MP4 simulation for Melbourne CBD.

Usage:
    python -m src.simulation.marauders_chronicle.render [--fps 20] [--hold 18] [--still 17]

Reads STOA's hourly CBD predictions from ``data/simulation/marauders_map/`` (written by
``src.simulation.marauders_map.predict``) and renders a clean, full-bleed 16:9 H.264 MP4
animation under ``data/simulation/marauders_chronicle/``. All KPI cards, accuracy metrics,
and sensor leaderboards live in the STOA web dashboard (``src.dashboard.build``) so the
video remains purely visual and uncluttered.

Visual encoding and data provenance:
- Every animated footprint trail, corridor flow-vein, and sensor halo is driven strictly
  by the registered STOA flow model predictions (``predictions.parquet``) and the
  Melbourne pedestrian count sensors (``sensors.json``) — no invented counts.
- Walkers spawn with probability proportional to ``sqrt(predicted_flow)`` so quieter
  streets remain legible alongside peak arteries; this square-root scaling is a visual
  encoding choice, not a model output.
- Prominent CBD locations are anchored to OpenStreetMap geometries (``transit``,
  ``ped_areas``, ``green``) with category-colored, collision-free scroll ribbons.
- Sensor pins are deduplicated by proximity so co-located poles do not double-stack;
  solid crimson rings show predicted volume and dashed teal rings show observed counts.
- Planning guidance, not a crowd-safety assessment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import LineCollection, PolyCollection
from scipy.ndimage import gaussian_filter

from src.ingest.common import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "simulation" / "marauders_map"
OUT_DIR = REPO_ROOT / "data" / "simulation" / "marauders_chronicle"
INTERIM = REPO_ROOT / "data" / "interim" / "osm" / "melbourne"

# Full-bleed 16:9 framing of the Hoddle Grid, Yarra Riverfront & Southbank (EPSG:28355).
# Width = 3360 m, Height = 1890 m -> exact 16:9 aspect ratio matching 1600 x 900 px.
VIEW = (319180.0, 5811860.0, 322540.0, 5813750.0)
W, H = 1600, 900  # 16:9 widescreen, both even for H.264
DPI = 100

# Vibrant, sunlit minimalist parchment & illuminated-atlas palette
PARCHMENT = np.array([0.980, 0.955, 0.895])  # Sunlit warm vellum (#FAF4E4)
PARCHMENT_EDGE = np.array([0.915, 0.855, 0.735])  # Warm golden-sand edge
CARD_BG = np.array([0.992, 0.976, 0.935])  # Crisp label fill
INK = np.array([0.135, 0.145, 0.210])  # Deep ink-indigo (#222536) for footprints
INK_SEPIA = np.array([0.235, 0.170, 0.125])  # Warm sepia for street lines
INK_MUTED = np.array([0.440, 0.375, 0.315])  # Soft warm slate-sepia
CRIMSON = np.array(
    [0.785, 0.180, 0.200]
)  # Vibrant crimson-ruby (#C82E33) for predictions
TEAL = np.array(
    [0.115, 0.455, 0.485]
)  # Rich atlas teal (#1D747C) for transit & observed
FOREST = np.array([0.245, 0.465, 0.295])  # Lush botanical green (#3F774B) for park pins
PARK_FILL = np.array([0.755, 0.855, 0.690])  # Vibrant sage-emerald parks (#C1DAB0)
PARK_EDGE = np.array([0.465, 0.615, 0.415])  # Crisp park outline
BLOCK_FILL = np.array([0.905, 0.835, 0.715])  # Warm sunlit sandstone blocks
MARIGOLD = np.array([0.925, 0.625, 0.175])  # Bright marigold flow wash (#ECA02D)
VERMILION = np.array([0.855, 0.305, 0.165])  # Vibrant coral-vermilion (#DA4E2A)
RUBY = np.array([0.640, 0.105, 0.215])  # Deep ruby peak flow (#A31B37)
GOLD_RULE = np.array([0.725, 0.595, 0.395])  # Burnished gold hairline rule

# Prominent CBD landmarks: (display_name, osm_layer, regex, mode, category, dx_m, dy_m, ha)
LANDMARKS: list[tuple[str, str, str, str | None, str, float, float, str]] = [
    (
        "Flinders Street Station",
        "transit",
        r"^Flinders Street$",
        "rail_station",
        "transit",
        70,
        -84,
        "left",
    ),
    (
        "Southern Cross Station",
        "transit",
        r"^Southern Cross Station$",
        "bus_stop",
        "transit",
        68,
        56,
        "left",
    ),
    (
        "Melbourne Central",
        "transit",
        r"Melbourne Central and State Library",
        None,
        "transit",
        78,
        62,
        "left",
    ),
    (
        "Parliament Station",
        "transit",
        r"Parliament Station",
        None,
        "transit",
        70,
        48,
        "left",
    ),
    (
        "Bourke Street Mall",
        "ped_areas",
        r"^Bourke Street Mall$",
        None,
        "civic",
        -78,
        56,
        "right",
    ),
    (
        "Queen Victoria Market",
        "transit",
        r"Stop 7: Queen Victoria Market",
        None,
        "civic",
        65,
        -24,
        "left",
    ),
    ("Town Hall", "transit", r"Stop 6: Town Hall", None, "civic", 74, 12, "left"),
    (
        "Flagstaff Gardens",
        "green",
        r"^Flagstaff Gardens$",
        None,
        "park",
        0,
        -102,
        "center",
    ),
    ("Treasury Gardens", "green", r"^Treasury Gardens$", None, "park", 58, -68, "left"),
    ("Birrarung Marr", "green", r"^Birrarung Marr$", None, "park", 0, -86, "center"),
    ("Southbank", "green", r"^Southbank Garden$", None, "park", -68, 54, "right"),
]

CATEGORY_COLORS: dict[str, np.ndarray] = {
    "transit": TEAL,
    "civic": VERMILION,
    "park": FOREST,
}

# Anatomical footprint silhouettes (unit length = 1, facing +x).
_SOLE_BASE = np.array(
    [
        [0.48, 0.00],
        [0.42, 0.19],
        [0.24, 0.25],
        [0.02, 0.18],
        [-0.08, 0.12],
        [-0.08, -0.12],
        [0.02, -0.18],
        [0.24, -0.25],
        [0.42, -0.19],
    ],
    dtype=float,
)
_HEEL_BASE = np.array(
    [
        [-0.18, 0.12],
        [-0.36, 0.14],
        [-0.48, 0.08],
        [-0.50, 0.00],
        [-0.48, -0.08],
        [-0.36, -0.14],
        [-0.18, -0.12],
    ],
    dtype=float,
)


def pick_font(preferred: tuple[str, ...], fallback: str = "serif") -> str:
    """Return the first available font family from `preferred`, else `fallback`."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return fallback


def day_phase(hour: float) -> str:
    """Human-readable phase of the day for the minimal clock badge."""
    h = hour % 24.0
    if h < 5.0:
        return "Late Night"
    if h < 7.0:
        return "Early Dawn"
    if h < 10.0:
        return "Morning Commute"
    if h < 12.0:
        return "Late Morning"
    if h < 14.5:
        return "Midday Rush"
    if h < 16.5:
        return "Afternoon"
    if h < 19.5:
        return "Evening Peak"
    if h < 22.0:
        return "Evening Dining"
    return "Nightfall"


def flow_vein_colors(intensity: np.ndarray) -> np.ndarray:
    """Map normalized [0, 1] flow intensity to a vibrant marigold -> vermilion -> ruby palette."""
    t = np.clip(intensity, 0.0, 1.0)
    rgb = np.empty((len(t), 3), dtype=float)
    low = t <= 0.55
    u_low = np.where(low, t / 0.55, 0.0)[:, None]
    u_high = np.where(~low, (t - 0.55) / 0.45, 0.0)[:, None]
    rgb_low = MARIGOLD * (1.0 - u_low) + VERMILION * u_low
    rgb_high = VERMILION * (1.0 - u_high) + RUBY * u_high
    rgb[:] = np.where(low[:, None], rgb_low, rgb_high)
    return rgb


def deduplicate_sensors(
    sensors: list[dict[str, Any]],
    view: tuple[float, float, float, float] = VIEW,
    min_dist_m: float = 28.0,
) -> list[dict[str, Any]]:
    """Filter sensors to `view` and deduplicate co-located poles for cleaner spacing."""
    margin = 55.0
    in_view = [
        s
        for s in sensors
        if (view[0] + margin) <= s["x"] <= (view[2] - margin)
        and (view[1] + margin) <= s["y"] <= (view[3] - margin)
    ]

    def priority(s: dict[str, Any]) -> tuple[int, float]:
        obs_count = sum(1 for v in s.get("observed", []) if v is not None)
        peak_pred = max(
            (v for v in s.get("predicted", []) if v is not None), default=0.0
        )
        return (1 if obs_count > 0 else 0, peak_pred)

    ordered = sorted(in_view, key=priority, reverse=True)
    kept: list[dict[str, Any]] = []
    for s in ordered:
        if any(math.hypot(s["x"] - k["x"], s["y"] - k["y"]) < min_dist_m for k in kept):
            continue
        kept.append(s)
    return kept


def find_landmarks(
    interim_dir: Path = INTERIM,
    view: tuple[float, float, float, float] = VIEW,
) -> list[dict[str, Any]]:
    """Locate prominent CBD landmarks from pinned OSM layers, strictly inside `view`."""
    pad_x, pad_y = 100.0, 80.0
    cache: dict[str, gpd.GeoDataFrame] = {}
    out: list[dict[str, Any]] = []
    for label, layer, pattern, mode, category, dx, dy, ha in LANDMARKS:
        if layer not in cache:
            path = interim_dir / f"{layer}.parquet"
            if not path.exists():
                continue
            cache[layer] = gpd.read_parquet(path)
        g = cache[layer]
        hit = g[
            g["name"].astype(str).str.contains(pattern, flags=re.IGNORECASE, regex=True)
        ]
        if mode and "mode" in hit.columns:
            hit = hit[hit["mode"].eq(mode)]
        if hit.empty:
            continue
        c = hit.geometry.centroid
        x, y = float(c.x.mean()), float(c.y.mean())
        if (view[0] + pad_x) <= x <= (view[2] - pad_x) and (view[1] + pad_y) <= y <= (
            view[3] - pad_y
        ):
            out.append(
                {
                    "name": label,
                    "category": category,
                    "x": x,
                    "y": y,
                    "dx": dx,
                    "dy": dy,
                    "ha": ha,
                }
            )
    return out


def footprint_polygons(
    xs: np.ndarray,
    ys: np.ndarray,
    angles_rad: np.ndarray,
    scale: float = 17.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build vectorized (N, V, 2) world-space sole and heel polygons for footprints."""
    c = np.cos(angles_rad)[:, None, None]
    s = np.sin(angles_rad)[:, None, None]
    rot = np.concatenate(
        [np.concatenate([c, -s], axis=2), np.concatenate([s, c], axis=2)],
        axis=1,
    )
    centers = np.stack([xs, ys], axis=1)[:, None, :]
    sole = np.einsum("nij,vj->nvi", rot, _SOLE_BASE * scale) + centers
    heel = np.einsum("nij,vj->nvi", rot, _HEEL_BASE * scale) + centers
    return sole, heel


def make_parchment(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a bright, warm sunlit-vellum full-bleed canvas with delicate paper grain."""
    grain = (
        0.25 * gaussian_filter(rng.normal(size=(height, width)), 1.0)
        + 0.40 * gaussian_filter(rng.normal(size=(height, width)), 9.0)
        + 0.55 * gaussian_filter(rng.normal(size=(height, width)), 44.0)
    )
    grain = (grain - grain.mean()) / max(grain.std(), 1e-6)

    yy, xx = np.mgrid[0:height, 0:width]
    u = xx / width - 0.5
    v = yy / height - 0.5
    radial = np.sqrt((u / 0.62) ** 2 + (v / 0.64) ** 2)
    vignette = np.clip((radial - 0.60) / 0.56, 0.0, 1.0) ** 1.9

    img = np.empty((height, width, 3), dtype=np.float32)
    for k in range(3):
        base = PARCHMENT[k] * (1.0 - vignette * 0.34) + PARCHMENT_EDGE[k] * (
            vignette * 0.34
        )
        img[..., k] = base + 0.0085 * grain
    return np.clip(img, 0.0, 1.0)


def render_base_ink(
    segs_in_view: gpd.GeoDataFrame,
    mean_flow: np.ndarray,
    buildings: list[np.ndarray],
    parks: list[np.ndarray],
    map_w: int,
    map_h: int,
) -> np.ndarray:
    """Render vibrant, spaced-out minimalist streets, emerald parks, and warm sandstone blocks."""
    fig = plt.figure(figsize=(map_w / DPI, map_h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(VIEW[0], VIEW[2])
    ax.set_ylim(VIEW[1], VIEW[3])
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    if parks:
        ax.add_collection(
            PolyCollection(
                parks,
                facecolors=[(*PARK_FILL, 0.72)],
                edgecolors=[(*PARK_EDGE, 0.42)],
                linewidths=0.6,
                zorder=1,
            )
        )

    if buildings:
        ax.add_collection(
            PolyCollection(
                buildings,
                facecolors=[(*BLOCK_FILL, 0.38)],
                edgecolors=[(*GOLD_RULE, 0.20)],
                linewidths=0.25,
                zorder=2,
            )
        )

    lines = [np.asarray(g.coords) for g in segs_in_view.geometry]
    q45 = float(np.quantile(mean_flow, 0.45)) if len(mean_flow) else 0.0
    q78 = float(np.quantile(mean_flow, 0.78)) if len(mean_flow) else 0.0
    q92 = float(np.quantile(mean_flow, 0.92)) if len(mean_flow) else 0.0

    tiers = [
        ((mean_flow > q45) & (mean_flow <= q78), 0.38, 0.22),
        ((mean_flow > q78) & (mean_flow <= q92), 0.72, 0.40),
        (mean_flow > q92, 1.15, 0.58),
    ]
    for mask, lw, alpha in tiers:
        subset = [ln for ln, keep in zip(lines, mask) if keep]
        if subset:
            ax.add_collection(
                LineCollection(
                    subset,
                    colors=[(*INK_SEPIA, alpha)],
                    linewidths=lw,
                    capstyle="round",
                    zorder=3,
                )
            )

    for inset, lw, alpha in ((18, 1.25, 0.78), (26, 0.50, 0.48)):
        dx = inset * (VIEW[2] - VIEW[0]) / map_w
        dy = inset * (VIEW[3] - VIEW[1]) / map_h
        x0, x1 = VIEW[0] + dx, VIEW[2] - dx
        y0, y1 = VIEW[1] + dy, VIEW[3] - dy
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            color=(*GOLD_RULE, alpha),
            lw=lw,
            zorder=5,
        )

    # Minimalist teal-and-gold compass rose in top-right corner
    cx = VIEW[2] - 125.0
    cy = VIEW[3] - 125.0
    r_outer, r_inner = 52.0, 13.0
    star = np.array(
        [
            [cx, cy + r_outer],
            [cx + r_inner, cy + r_inner],
            [cx + r_outer * 0.72, cy],
            [cx + r_inner, cy - r_inner],
            [cx, cy - r_outer * 0.72],
            [cx - r_inner, cy - r_inner],
            [cx - r_outer * 0.72, cy],
            [cx - r_inner, cy + r_inner],
        ]
    )
    ax.add_patch(
        mpatches.Polygon(
            star,
            closed=True,
            facecolor=(*TEAL, 0.16),
            edgecolor=(*TEAL, 0.72),
            linewidth=0.75,
            zorder=6,
        )
    )
    ax.plot(
        [cx, cx],
        [cy - r_outer * 0.85, cy + r_outer * 1.05],
        color=(*TEAL, 0.72),
        lw=0.65,
        zorder=6,
    )
    ax.plot(
        [cx - r_outer * 0.85, cx + r_outer * 0.85],
        [cy, cy],
        color=(*TEAL, 0.72),
        lw=0.65,
        zorder=6,
    )
    ax.text(
        cx,
        cy + r_outer + 24.0,
        "N",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=(*TEAL, 0.90),
        zorder=6,
    )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy().astype(np.float32) / 255.0
    plt.close(fig)
    return rgba


class MarauderWalkers:
    """Animated left-right footprint walkers driven by STOA's hourly segment predictions."""

    STRIDE_M = 21.0
    SPEED_M = 4.4
    LIFE_FRAMES = 18
    WALK_FRAMES = 110

    def __init__(
        self, segs: gpd.GeoDataFrame, flow: np.ndarray, seed: int = 19
    ) -> None:
        self.rng = np.random.default_rng(seed)
        keep = np.array(
            [
                (VIEW[0] - 40) <= g.centroid.x <= (VIEW[2] + 40)
                and (VIEW[1] - 40) <= g.centroid.y <= (VIEW[3] + 40)
                for g in segs.geometry
            ]
        )
        self.idx = np.flatnonzero(keep)
        self.geoms = np.array(list(segs.geometry.iloc[self.idx]), dtype=object)
        self.lengths = np.maximum(shapely.length(self.geoms), 1.0)
        self.flow = np.nan_to_num(flow[self.idx], nan=0.0)

        pts0 = shapely.get_point(self.geoms, 0)
        pts1 = shapely.get_point(self.geoms, -1)
        self.k0 = list(
            zip(np.round(shapely.get_x(pts0), 1), np.round(shapely.get_y(pts0), 1))
        )
        self.k1 = list(
            zip(np.round(shapely.get_x(pts1), 1), np.round(shapely.get_y(pts1), 1))
        )
        self.adj: dict[tuple[float, float], list[tuple[int, int]]] = {}
        for i, (a, b) in enumerate(zip(self.k0, self.k1)):
            self.adj.setdefault(a, []).append((i, 1))
            self.adj.setdefault(b, []).append((i, -1))

        self.seg = np.empty(0, dtype=int)
        self.dist = np.empty(0, dtype=float)
        self.dir = np.empty(0, dtype=int)
        self.age = np.empty(0, dtype=int)
        self.since = np.empty(0, dtype=float)
        self.side = np.empty(0, dtype=int)

        self.fx = np.empty(0, dtype=float)
        self.fy = np.empty(0, dtype=float)
        self.fa = np.empty(0, dtype=float)
        self.fage = np.empty(0, dtype=int)
        self.q = self.flow[:, 0]

    def _heading(
        self, seg: np.ndarray, dist: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        a = shapely.line_interpolate_point(
            self.geoms[seg], np.clip(dist, 0.0, self.lengths[seg])
        )
        b = shapely.line_interpolate_point(
            self.geoms[seg], np.clip(dist + direction * 2.0, 0.0, self.lengths[seg])
        )
        dx = shapely.get_x(b) - shapely.get_x(a)
        dy = shapely.get_y(b) - shapely.get_y(a)
        return np.arctan2(dy * direction, dx * direction)

    def step(self, t_hours: float, spawn_per_frame: float) -> None:
        h0 = math.floor(t_hours) % 24
        h1 = (h0 + 1) % 24
        frac = t_hours - math.floor(t_hours)
        self.q = (1.0 - frac) * self.flow[:, h0] + frac * self.flow[:, h1]

        if spawn_per_frame > 0.0:
            n_new = int(self.rng.poisson(spawn_per_frame))
            if n_new > 0 and self.q.sum() > 0:
                w = np.sqrt(np.maximum(self.q, 0.0))
                probs = w / w.sum()
                s = self.rng.choice(len(w), size=n_new, p=probs)
                d = self.rng.choice([-1, 1], size=n_new)
                start = self.rng.uniform(0.0, self.lengths[s])
                self.seg = np.r_[self.seg, s]
                self.dist = np.r_[self.dist, start]
                self.dir = np.r_[self.dir, d]
                self.age = np.r_[self.age, np.zeros(n_new, dtype=int)]
                self.since = np.r_[
                    self.since, self.rng.uniform(0.0, self.STRIDE_M, n_new)
                ]
                self.side = np.r_[self.side, self.rng.choice([-1, 1], size=n_new)]

        self.dist = self.dist + self.dir * self.SPEED_M
        self.age += 1
        self.since += self.SPEED_M

        off = (self.dist < 0.0) | (self.dist > self.lengths[self.seg])
        for k in np.flatnonzero(off):
            i, d = int(self.seg[k]), int(self.dir[k])
            node = self.k1[i] if d == 1 else self.k0[i]
            opts = [(j, e) for j, e in self.adj.get(node, []) if j != i]
            if not opts:
                self.age[k] = self.WALK_FRAMES + 1
                continue
            js = np.array([j for j, _ in opts], dtype=int)
            es = np.array([e for _, e in opts], dtype=int)
            h_in = self._heading(
                np.array([i]),
                np.array([self.lengths[i] if d == 1 else 0.0]),
                np.array([d]),
            )[0]
            starts = np.where(es == 1, 0.0, self.lengths[js])
            h_out = self._heading(js, starts, es)
            straight = np.clip(np.cos(h_out - h_in), 0.0, 1.0)
            weights = np.maximum(self.q[js], 1.0) * (0.15 + 0.85 * straight**2)
            choice = int(self.rng.choice(len(js), p=weights / weights.sum()))
            self.seg[k], self.dir[k], self.dist[k] = (
                js[choice],
                es[choice],
                starts[choice],
            )

        alive = self.age < self.WALK_FRAMES
        self.seg = self.seg[alive]
        self.dist = self.dist[alive]
        self.dir = self.dir[alive]
        self.age = self.age[alive]
        self.since = self.since[alive]
        self.side = self.side[alive]

        self.fage += 1
        keep_f = self.fage < self.LIFE_FRAMES
        self.fx = self.fx[keep_f]
        self.fy = self.fy[keep_f]
        self.fa = self.fa[keep_f]
        self.fage = self.fage[keep_f]

        stepped = self.since >= self.STRIDE_M
        if stepped.any():
            s = self.seg[stepped]
            dd = self.dist[stepped]
            d = self.dir[stepped]
            pts = shapely.line_interpolate_point(
                self.geoms[s], np.clip(dd, 0.0, self.lengths[s])
            )
            ang = self._heading(s, dd, d)
            side = self.side[stepped]
            lateral = 4.5
            self.fx = np.r_[self.fx, shapely.get_x(pts) - np.sin(ang) * lateral * side]
            self.fy = np.r_[self.fy, shapely.get_y(pts) + np.cos(ang) * lateral * side]
            self.fa = np.r_[self.fa, ang]
            self.fage = np.r_[self.fage, np.zeros(int(stepped.sum()), dtype=int)]
            self.side[stepped] *= -1
            self.since[stepped] = 0.0


def draw_scroll_ribbon(
    ax: plt.Axes,
    x: float,
    y: float,
    dx: float,
    dy: float,
    label: str,
    category: str,
    ha: str,
    font_serif: str,
) -> None:
    """Draw a vibrant, category-colored pin + scroll banner for a prominent location."""
    accent = CATEGORY_COLORS.get(category, TEAL)
    ax.plot(
        [x, x + dx * 0.78], [y, y + dy * 0.78], color=(*accent, 0.55), lw=0.85, zorder=7
    )
    ax.plot(
        x,
        y,
        marker="D",
        ms=6.8,
        mfc=(*CARD_BG, 0.98),
        mec=(*accent, 0.95),
        mew=1.2,
        zorder=8,
    )
    ax.plot(x, y, marker="D", ms=2.6, mfc=(*accent, 0.98), mec="none", zorder=9)
    ax.text(
        x + dx,
        y + dy,
        label,
        fontfamily=font_serif,
        fontsize=9.4,
        fontweight="bold",
        color=(*INK, 0.96),
        ha=ha,
        va="center",
        bbox={
            "boxstyle": "round,pad=0.30,rounding_size=0.20",
            "facecolor": (*CARD_BG, 0.94),
            "edgecolor": (*accent, 0.75),
            "linewidth": 0.95,
        },
        zorder=10,
    )


def render(
    data_dir: Path = DATA_DIR,
    out_dir: Path = OUT_DIR,
    fps: int = 20,
    hold: int = 18,
    still_hour: float | None = None,
) -> list[Path]:
    """Render the clean, full-bleed STOA simulation MP4 (no KPIs or metric panels)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "meta.json").read_text())
    segs = gpd.read_parquet(data_dir / "segments.parquet")
    pred = pd.read_parquet(data_dir / "predictions.parquet")
    flow = (
        pred.pivot(index="segment_id", columns="hour", values="predicted")
        .reindex(segs["segment_id"])
        .to_numpy(dtype=float)
    )
    raw_sensors = json.loads((data_dir / "sensors.json").read_text())
    sensors = deduplicate_sensors(raw_sensors, VIEW, min_dist_m=28.0)
    landmarks = find_landmarks(INTERIM, VIEW)

    font_serif = pick_font(("Baskerville", "Palatino", "Georgia", "Times New Roman"))

    bld = gpd.read_parquet(INTERIM / "buildings.parquet", columns=["geometry"])
    bld = bld.cx[VIEW[0] : VIEW[2], VIEW[1] : VIEW[3]]
    bld_polys = [
        np.asarray(g.exterior.coords)
        for g in bld.geometry.explode(index_parts=False)
        if g.geom_type == "Polygon"
    ]
    parks_path = INTERIM / "green.parquet"
    park_polys: list[np.ndarray] = []
    if parks_path.exists():
        grn = gpd.read_parquet(parks_path, columns=["geometry"])
        grn = grn.cx[VIEW[0] : VIEW[2], VIEW[1] : VIEW[3]]
        park_polys = [
            np.asarray(g.exterior.coords)
            for g in grn.geometry.explode(index_parts=False)
            if g.geom_type == "Polygon"
        ]

    rng = np.random.default_rng(11)
    paper = make_parchment(W, H, rng)

    in_view = segs.cx[VIEW[0] : VIEW[2], VIEW[1] : VIEW[3]]
    mean_flow_in = np.nan_to_num(flow.mean(axis=1))[in_view.index]
    base_ink = render_base_ink(in_view, mean_flow_in, bld_polys, park_polys, W, H)

    vein_thresh = float(np.quantile(mean_flow_in, 0.86)) if len(mean_flow_in) else 0.0
    vein_mask = mean_flow_in >= vein_thresh
    vein_indices = in_view.index.to_numpy()[vein_mask]
    vein_lines = [
        np.asarray(g.coords) for g in in_view.geometry.iloc[np.flatnonzero(vein_mask)]
    ]
    vein_flow = np.nan_to_num(flow[vein_indices], nan=0.0)
    vein_peak = float(np.percentile(vein_flow, 98)) if vein_flow.size else 1.0

    # Full-bleed 16:9 canvas (no KPI sidebar)
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(VIEW[0], VIEW[2])
    ax.set_ylim(VIEW[1], VIEW[3])
    ax.axis("off")

    ext = (VIEW[0], VIEW[2], VIEW[1], VIEW[3])
    ax.imshow(paper, extent=ext, origin="upper", aspect="auto", zorder=0)
    ax.imshow(base_ink, extent=ext, origin="upper", aspect="auto", zorder=1)

    vein_lc = LineCollection(
        vein_lines,
        colors=[(*MARIGOLD, 0.0)],
        linewidths=2.8,
        capstyle="round",
        zorder=2,
    )
    ax.add_collection(vein_lc)

    sole_pc = PolyCollection([], facecolors=[(*INK, 0.0)], edgecolors="none", zorder=4)
    heel_pc = PolyCollection([], facecolors=[(*INK, 0.0)], edgecolors="none", zorder=4)
    ax.add_collection(sole_pc)
    ax.add_collection(heel_pc)

    sx = np.array([s["x"] for s in sensors], dtype=float)
    sy = np.array([s["y"] for s in sensors], dtype=float)
    sensor_peak = max((max(s["predicted"]) for s in sensors), default=1.0)
    ax.plot(
        sx,
        sy,
        "o",
        ms=3.8,
        mfc=(*CRIMSON, 0.95),
        mec=(*CARD_BG, 0.95),
        mew=0.6,
        zorder=6,
    )
    ring_pred = ax.scatter(
        sx,
        sy,
        s=10,
        facecolors=[(*CRIMSON, 0.14)],
        edgecolors=[(*CRIMSON, 0.90)],
        linewidths=1.15,
        zorder=5,
    )
    ring_obs = ax.scatter(
        sx,
        sy,
        s=10,
        facecolors="none",
        edgecolors=[(*TEAL, 0.85)],
        linewidths=0.95,
        linestyles=(0, (2.5, 2.0)),
        zorder=5,
    )
    pulse_ring = ax.scatter(
        sx[:3],
        sy[:3],
        s=10,
        facecolors="none",
        edgecolors=[(*CRIMSON, 0.0)],
        linewidths=1.45,
        zorder=6,
    )

    for lm in landmarks:
        draw_scroll_ribbon(
            ax,
            lm["x"],
            lm["y"],
            lm["dx"],
            lm["dy"],
            lm["name"],
            lm["category"],
            lm["ha"],
            font_serif,
        )

    # Minimal floating title + live clock badge (top-left only; zero KPIs/metrics)
    date_label = pd.Timestamp(meta.get("date", "2026-11-06")).strftime("%A, %-d %B %Y")
    badge_txt = ax.text(
        0.028,
        0.955,
        f"STOA  ·  00:00  ·  Late Night\nMelbourne CBD  ·  {date_label}",
        transform=ax.transAxes,
        fontfamily=font_serif,
        fontsize=11.5,
        fontweight="bold",
        color=(*INK, 0.96),
        va="top",
        ha="left",
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.55,rounding_size=0.35",
            "facecolor": (*CARD_BG, 0.88),
            "edgecolor": (*GOLD_RULE, 0.65),
            "linewidth": 0.9,
        },
        zorder=15,
    )

    # Whisper-thin bottom caption (map key + Core Rule 7 planning guidance)
    ax.text(
        0.5,
        0.026,
        (
            "Footprints & amber–ruby veins: predicted flow   ·   "
            "Crimson ring: predicted   ·   Teal dashed: observed   ·   "
            "Planning guidance, not a crowd-safety assessment"
        ),
        transform=ax.transAxes,
        fontfamily=font_serif,
        fontsize=8.2,
        style="italic",
        color=(*INK_MUTED, 0.90),
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.32,rounding_size=0.25",
            "facecolor": (*CARD_BG, 0.82),
            "edgecolor": (*GOLD_RULE, 0.40),
            "linewidth": 0.6,
        },
        zorder=15,
    )

    walkers = MarauderWalkers(segs, flow)
    totals = np.nansum(flow, axis=0)
    totals_max = max(float(totals.max()), 1.0)
    spawn_max = 18.0
    total_frames = 24 * hold

    def spawn_rate(t_hr: float) -> float:
        h0 = min(int(t_hr), 23)
        h1 = (h0 + 1) % 24
        f = t_hr - h0
        share = ((1.0 - f) * totals[h0] + f * totals[h1]) / totals_max
        return spawn_max * (0.14 + 0.86 * share**0.85)

    def render_frame(frame_idx: int) -> np.ndarray:
        t_hr = (frame_idx / hold) % 24.0
        walkers.step(t_hr, spawn_rate(t_hr))

        h0 = min(int(t_hr), 23)
        h1 = (h0 + 1) % 24
        frac = t_hr - h0

        if vein_flow.size:
            vq = (1.0 - frac) * vein_flow[:, h0] + frac * vein_flow[:, h1]
            intensity = np.clip(np.log1p(vq) / np.log1p(vein_peak), 0.0, 1.0) ** 1.45
            vein_rgba = np.zeros((len(vq), 4), dtype=float)
            vein_rgba[:, :3] = flow_vein_colors(intensity)
            vein_rgba[:, 3] = 0.46 * intensity
            vein_lc.set_colors(vein_rgba)
            vein_lc.set_linewidths(1.3 + 2.8 * intensity)

        n_fp = len(walkers.fx)
        if n_fp > 0:
            age_norm = walkers.fage / walkers.LIFE_FRAMES
            fp_alpha = 0.92 * (1.0 - age_norm) ** 1.5
            sole_v, heel_v = footprint_polygons(
                walkers.fx, walkers.fy, walkers.fa, scale=17.5
            )
            fp_col = np.c_[np.tile(INK, (n_fp, 1)), fp_alpha]
            sole_pc.set_verts(sole_v)
            heel_pc.set_verts(heel_v)
            sole_pc.set_facecolor(fp_col)
            heel_pc.set_facecolor(fp_col)
        else:
            sole_pc.set_verts([])
            heel_pc.set_verts([])

        sp = np.array(
            [
                (1.0 - frac) * s["predicted"][h0] + frac * s["predicted"][h1]
                for s in sensors
            ]
        )
        so = np.array(
            [
                np.nan
                if (s["observed"][h0] is None or s["observed"][h1] is None)
                else (1.0 - frac) * s["observed"][h0] + frac * s["observed"][h1]
                for s in sensors
            ]
        )

        def ring_area(vals: np.ndarray) -> np.ndarray:
            r_pts = 3.5 + 14.0 * np.sqrt(
                np.clip(np.nan_to_num(vals), 0.0, None) / sensor_peak
            )
            return np.pi * r_pts**2

        ring_pred.set_sizes(ring_area(sp))
        ring_obs.set_sizes(np.where(np.isnan(so), 0.0, ring_area(so)))

        top3_idx = np.argsort(-sp)[:3]
        pulse_phase = (frame_idx % 20) / 20.0
        pulse_ring.set_offsets(np.c_[sx[top3_idx], sy[top3_idx]])
        pulse_r = 10.5 + 19.0 * pulse_phase
        pulse_ring.set_sizes(np.full(len(top3_idx), np.pi * pulse_r**2))
        pulse_ring.set_edgecolors([(*CRIMSON, 0.62 * (1.0 - pulse_phase))])

        hour_int = int(t_hr) % 24
        minute_int = (int((t_hr - int(t_hr)) * 60) // 5) * 5
        badge_txt.set_text(
            f"STOA  ·  {hour_int:02d}:{minute_int:02d}  ·  {day_phase(t_hr)}\n"
            f"Melbourne CBD  ·  {date_label}"
        )

        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    if still_hour is not None:
        target = int(still_hour * hold)
        for i in range(max(0, target - 60), target):
            walkers.step((i / hold) % 24.0, spawn_rate((i / hold) % 24.0))
        img = render_frame(target)
        still_path = out_dir / f"still_{int(still_hour):02d}h.png"
        plt.imsave(still_path, img)
        plt.close(fig)
        print(f"wrote {still_path}")
        return [still_path]

    for _ in range(40):
        walkers.step(0.0, spawn_rate(0.0))

    import imageio.v2 as imageio

    mp4_path = out_dir / "marauders_chronicle.mp4"
    writer = imageio.get_writer(
        str(mp4_path), fps=fps, codec="libx264", quality=8, macro_block_size=2
    )
    for i in range(total_frames):
        writer.append_data(render_frame(i))
        if i % 60 == 0:
            print(f"frame {i}/{total_frames}", flush=True)
    writer.close()
    plt.close(fig)
    print(f"wrote {mp4_path}")
    return [mp4_path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the full-bleed STOA MP4 simulation (metrics live in the dashboard)."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--hold", type=int, default=18, help="Frames per simulated hour"
    )
    parser.add_argument(
        "--still", type=float, default=None, help="Render a single frame at hour"
    )
    args = parser.parse_args()
    render(
        data_dir=args.data_dir,
        out_dir=args.out,
        fps=args.fps,
        hold=args.hold,
        still_hour=args.still,
    )


if __name__ == "__main__":
    main()
