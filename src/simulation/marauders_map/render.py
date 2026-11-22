"""The Marauder's Map: Stoa's predicted foot traffic in the CBD for one day, as a video and GIF.

Usage: python -m src.simulation.marauders_map.render [--fps 20] [--still 13]

Reads data/simulation/marauders_map/ (written by `predict`) and draws parchment, the street
network in ink, the prominent places, the count sensors, and footprints that walk the
streets. Writes marauders_map.mp4 and marauders_map.gif beside it.

What is data and what is drawing:
- Every footprint is drawn from the flow model's predictions (predictions.parquet). Walkers
  appear on a segment with a chance proportional to the square root of its predicted flow
  that hour, then walk on along the connected paths, preferring the busiest and straightest.
  The square root keeps quiet streets visible; it is a drawing choice, not a prediction.
- More footprints means more people. The footprints do not count anyone: the numbers on screen
  are the sensors' predicted and counted hours, which come from the model and the counts.
- Sensor rings: solid is Stoa's prediction, dashed is what the sensor counted. The model
  trained on these sensors, so agreement is a consistency check, not a test.
- Planning guidance, not a crowd-safety assessment (CLAUDE.md, core rule 7).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import EllipseCollection, LineCollection
from scipy.ndimage import gaussian_filter

from src.ingest.common import REPO_ROOT

DIR = REPO_ROOT / "data" / "simulation" / "marauders_map"
INTERIM = REPO_ROOT / "data" / "interim" / "osm" / "melbourne"

# The view: the Hoddle grid and its edges, in the site CRS (EPSG:28355).
VIEW = (319650.0, 5811550.0, 322400.0, 5813850.0)
W, H = 1560, 1300  # pixels; both even for H.264
DPI = 100

PARCHMENT = np.array([0.905, 0.835, 0.665])
INK = np.array([0.17, 0.10, 0.05])
FONT = "Apple Chancery"

# Prominent places, found by name in the OSM layers so each position has a source.
LANDMARKS = [
    ("Flinders Street Station", "transit", r"^Flinders Street$", "rail_station"),
    ("Southern Cross Station", "transit", r"^Southern Cross Station$", "bus_stop"),
    ("Melbourne Central", "transit", r"Melbourne Central and State Library", None),
    ("Parliament", "transit", r"Parliament Station", None),
    ("Bourke Street Mall", "ped_areas", r"^Bourke Street Mall$", None),
    ("Queen Victoria Market", "transit", r"Stop 7: Queen Victoria Market", None),
    ("Town Hall", "transit", r"Stop 6: Town Hall", None),
    ("Flagstaff Gardens", "green", r"^Flagstaff Gardens$", None),
    ("Treasury Gardens", "green", r"^Treasury Gardens$", None),
    ("Fitzroy Gardens", "green", r"^Fitzroy Gardens$", None),
    ("Birrarung Marr", "green", r"^Birrarung Marr$", None),
    ("Southbank", "green", r"^Southbank Garden$", None),
]


def find_landmarks() -> list[dict]:
    out = []
    for label, layer, pattern, mode in LANDMARKS:
        g = gpd.read_parquet(INTERIM / f"{layer}.parquet")
        hit = g[
            g["name"].astype(str).str.contains(pattern, flags=re.IGNORECASE, regex=True)
        ]
        if mode and "mode" in hit.columns:
            hit = hit[hit["mode"].eq(mode)]
        if hit.empty:
            continue
        c = hit.geometry.centroid
        x, y = float(c.x.mean()), float(c.y.mean())
        if VIEW[0] <= x <= VIEW[2] and VIEW[1] <= y <= VIEW[3]:
            out.append({"name": label, "x": x, "y": y})
    return out


# --------------------------------------------------------------------------------------------
# Parchment and ink
# --------------------------------------------------------------------------------------------


def parchment(rng: np.random.Generator) -> np.ndarray:
    """Aged paper: fibre noise, blotches, a burnt vignette and two folds."""
    n = rng.normal(size=(H, W))
    tex = (
        0.35 * gaussian_filter(n, 1.2)
        + 0.5 * gaussian_filter(rng.normal(size=(H, W)), 6)
        + 0.9 * gaussian_filter(rng.normal(size=(H, W)), 28)
        + 1.2 * gaussian_filter(rng.normal(size=(H, W)), 110)
    )
    tex = (tex - tex.mean()) / tex.std()
    yy, xx = np.mgrid[0:H, 0:W]
    u, v = xx / W - 0.5, yy / H - 0.5
    r = np.sqrt((u / 0.62) ** 2 + (v / 0.62) ** 2)
    burn = np.clip((r - 0.62) / 0.55, 0, 1) ** 1.6
    edge = np.minimum.reduce([xx, W - xx, yy, H - yy]).astype(float)
    ragged = gaussian_filter(rng.normal(size=(H, W)), 14) * 90
    burn = np.maximum(burn, np.clip(1 - (edge + ragged) / 70, 0, 1) ** 1.5)
    folds = np.zeros((H, W))
    for pos, vertical in ((0.5, True), (0.5, False)):
        d = np.abs((u if vertical else v) - (pos - 0.5))
        folds += np.exp(-((d * (W if vertical else H)) ** 2) / 5.0)
    img = np.empty((H, W, 3))
    for k in range(3):
        img[..., k] = PARCHMENT[k] + 0.028 * tex
    img -= 0.10 * burn[..., None] * np.array([0.9, 1.1, 1.5])
    img -= 0.035 * folds[..., None]
    return np.clip(img, 0, 1)


def ink_layer(
    segs: gpd.GeoDataFrame, mean_flow: np.ndarray, buildings: list[np.ndarray]
) -> np.ndarray:
    """Streets, buildings and the border, drawn in ink on a transparent background."""
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(VIEW[0], VIEW[2])
    ax.set_ylim(VIEW[1], VIEW[3])
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    from matplotlib.collections import PolyCollection

    ax.add_collection(
        PolyCollection(
            buildings,
            facecolors=[(*INK, 0.075)],
            edgecolors=[(*INK, 0.30)],
            linewidths=0.35,
        )
    )
    lines = [np.asarray(g.coords) for g in segs.geometry]
    busy = mean_flow > np.quantile(mean_flow, 0.80)
    for sel, lw, a in ((~busy, 0.45, 0.42), (busy, 0.9, 0.62)):
        ax.add_collection(
            LineCollection(
                [ln for ln, s in zip(lines, sel) if s],
                colors=[(*INK, a)],
                linewidths=lw,
                capstyle="round",
            )
        )
    # Double-ruled border with a small notch at each corner.
    m1, m2 = 26, 36
    for m, lw in ((m1, 1.6), (m2, 0.6)):
        x0, x1 = (
            VIEW[0] + m * (VIEW[2] - VIEW[0]) / W,
            VIEW[2] - m * (VIEW[2] - VIEW[0]) / W,
        )
        y0, y1 = (
            VIEW[1] + m * (VIEW[3] - VIEW[1]) / H,
            VIEW[3] - m * (VIEW[3] - VIEW[1]) / H,
        )
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=(*INK, 0.85), lw=lw)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img.astype(np.float32) / 255.0


# --------------------------------------------------------------------------------------------
# Walkers
# --------------------------------------------------------------------------------------------


class Walkers:
    """Footprint walkers on the CBD segments, driven by the predicted flow of each hour."""

    STRIDE = 11.0  # metres between footprints
    SPEED = 3.0  # metres per frame
    LIFE = 14  # frames a footprint stays visible
    WALK = 120  # frames a walker lives

    def __init__(self, segs: gpd.GeoDataFrame, flow: np.ndarray, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        keep = np.array(
            [
                VIEW[0] - 50 <= g.centroid.x <= VIEW[2] + 50
                and VIEW[1] - 50 <= g.centroid.y <= VIEW[3] + 50
                for g in segs.geometry
            ]
        )
        self.idx = np.flatnonzero(keep)
        self.geoms = np.array(list(segs.geometry.iloc[self.idx]), dtype=object)
        self.len = shapely.length(self.geoms)
        self.flow = flow[self.idx]  # (n, 24)
        pts0 = shapely.get_point(self.geoms, 0)
        pts1 = shapely.get_point(self.geoms, -1)

        def _key(p):
            return list(
                zip(np.round(shapely.get_x(p), 1), np.round(shapely.get_y(p), 1))
            )

        self.k0, self.k1 = _key(pts0), _key(pts1)
        self.at: dict[tuple, list[tuple[int, int]]] = {}
        for i, (a, b) in enumerate(zip(self.k0, self.k1)):
            self.at.setdefault(a, []).append(
                (i, 1)
            )  # leave from the start, walking forward
            self.at.setdefault(b, []).append((i, -1))
        self.seg = np.empty(0, int)
        self.dist = np.empty(0)
        self.dir = np.empty(0, int)
        self.age = np.empty(0, int)
        self.since = np.empty(0)
        self.side = np.empty(0, int)
        self.fx = np.empty(0)
        self.fy = np.empty(0)
        self.fa = np.empty(0)
        self.fage = np.empty(0, int)
        self.q = self.flow[:, 0]

    def _heading(self, seg: np.ndarray, dist: np.ndarray, d: np.ndarray) -> np.ndarray:
        a = shapely.line_interpolate_point(
            self.geoms[seg], np.clip(dist, 0, self.len[seg])
        )
        b = shapely.line_interpolate_point(
            self.geoms[seg], np.clip(dist + d * 1.5, 0, self.len[seg])
        )
        dx = shapely.get_x(b) - shapely.get_x(a)
        dy = shapely.get_y(b) - shapely.get_y(a)
        return np.arctan2(dy * d, dx * d)

    def step(self, t_hours: float, spawn_per_frame: float) -> None:
        h0 = math.floor(t_hours) % 24
        h1 = (h0 + 1) % 24
        f = t_hours - math.floor(t_hours)
        self.q = (1 - f) * self.flow[:, h0] + f * self.flow[:, h1]
        # Spawn.
        n_new = self.rng.poisson(spawn_per_frame)
        if n_new:
            w = np.sqrt(self.q)
            s = self.rng.choice(len(w), size=n_new, p=w / w.sum())
            d = self.rng.choice([-1, 1], size=n_new)
            start = self.rng.uniform(0, self.len[s])
            self.seg = np.r_[self.seg, s]
            self.dist = np.r_[self.dist, start]
            self.dir = np.r_[self.dir, d]
            self.age = np.r_[self.age, np.zeros(n_new, int)]
            self.since = np.r_[self.since, self.rng.uniform(0, self.STRIDE, n_new)]
            self.side = np.r_[self.side, self.rng.choice([-1, 1], size=n_new)]
        # Move.
        self.dist = self.dist + self.dir * self.SPEED
        self.age += 1
        self.since += self.SPEED
        # Walkers that reach a segment end continue on a connected path.
        off = (self.dist < 0) | (self.dist > self.len[self.seg])
        for k in np.flatnonzero(off):
            i, d = self.seg[k], self.dir[k]
            node = self.k1[i] if d == 1 else self.k0[i]
            opts = [(j, e) for j, e in self.at.get(node, []) if j != i]
            if not opts:
                self.age[k] = 10**6
                continue
            js = np.array([j for j, _ in opts])
            es = np.array([e for _, e in opts])
            heading0 = self._heading(
                np.array([i]), np.array([self.len[i] if d == 1 else 0.0]), np.array([d])
            )[0]
            starts = np.where(es == 1, 0.0, self.len[js])
            h1_ = self._heading(js, starts, es)
            straight = np.cos(h1_ - heading0)
            weight = np.maximum(self.q[js], 1.0) * (
                0.15 + 0.85 * np.clip(straight, 0, 1) ** 2
            )
            c = self.rng.choice(len(js), p=weight / weight.sum())
            self.seg[k], self.dir[k], self.dist[k] = js[c], es[c], starts[c]
        alive = self.age < self.WALK
        for name in ("seg", "dist", "dir", "age", "since", "side"):
            setattr(self, name, getattr(self, name)[alive])
        # Footprints.
        self.fage += 1
        keep = self.fage < self.LIFE
        self.fx, self.fy, self.fa, self.fage = (
            self.fx[keep],
            self.fy[keep],
            self.fa[keep],
            self.fage[keep],
        )
        st = self.since >= self.STRIDE
        if st.any():
            s, dd, d = self.seg[st], self.dist[st], self.dir[st]
            p = shapely.line_interpolate_point(
                self.geoms[s], np.clip(dd, 0, self.len[s])
            )
            ang = self._heading(s, dd, d)
            side = self.side[st]
            self.fx = np.r_[self.fx, shapely.get_x(p) - np.sin(ang) * 1.8 * side]
            self.fy = np.r_[self.fy, shapely.get_y(p) + np.cos(ang) * 1.8 * side]
            self.fa = np.r_[self.fa, np.degrees(ang)]
            self.fage = np.r_[self.fage, np.zeros(st.sum(), int)]
            self.side[st] *= -1
            self.since[st] = 0.0


# --------------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------------


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--still", type=float, default=None, help="render one frame at this hour"
    )
    ap.add_argument("--hold", type=int, default=20, help="frames per hour")
    ap.add_argument("--out", default=str(DIR))
    args = ap.parse_args()
    out = Path(args.out)

    meta = json.loads((DIR / "meta.json").read_text())
    segs = gpd.read_parquet(DIR / "segments.parquet")
    pred = pd.read_parquet(DIR / "predictions.parquet")
    flow = (
        pred.pivot(index="segment_id", columns="hour", values="predicted")
        .reindex(segs["segment_id"])
        .to_numpy(dtype=float)
    )
    sensors = json.loads((DIR / "sensors.json").read_text())
    sensors = [
        s for s in sensors if VIEW[0] < s["x"] < VIEW[2] and VIEW[1] < s["y"] < VIEW[3]
    ]
    marks = find_landmarks()

    rng = np.random.default_rng(3)
    bld = gpd.read_parquet(INTERIM / "buildings.parquet", columns=["geometry"])
    bld = bld.cx[VIEW[0] : VIEW[2], VIEW[1] : VIEW[3]]
    polys = [
        np.asarray(g.exterior.coords)
        for g in bld.geometry.explode(index_parts=False)
        if g.geom_type == "Polygon"
    ]
    paper = parchment(rng)
    in_view = segs.cx[VIEW[0] : VIEW[2], VIEW[1] : VIEW[3]]
    ink = ink_layer(in_view, np.nan_to_num(flow.mean(axis=1))[in_view.index], polys)

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(VIEW[0], VIEW[2])
    ax.set_ylim(VIEW[1], VIEW[3])
    ax.axis("off")
    ext = (VIEW[0], VIEW[2], VIEW[1], VIEW[3])
    ax.imshow(paper, extent=ext, origin="upper", aspect="auto", zorder=0)
    ink_im = ax.imshow(ink, extent=ext, origin="upper", aspect="auto", zorder=1)

    halo = [pe.withStroke(linewidth=3.2, foreground=(*PARCHMENT, 0.95))]
    # Prominent places: a small ink diamond and a label.
    for m in marks:
        ax.plot(
            m["x"],
            m["y"],
            marker="D",
            ms=5.5,
            mfc=(*INK, 0.9),
            mec=(*INK, 1),
            mew=0.6,
            zorder=6,
        )
        ax.plot(
            m["x"],
            m["y"],
            marker="o",
            ms=17,
            mfc="none",
            mec=(*INK, 0.65),
            mew=0.7,
            zorder=6,
        )
        ax.text(
            m["x"] + 38,
            m["y"] + 24,
            m["name"],
            fontname=FONT,
            fontsize=15.5,
            color=(*INK, 0.95),
            path_effects=halo,
            zorder=7,
        )
    mark_xy = np.array([[m["x"], m["y"]] for m in marks])
    # Sensors: a dot, a solid ring for the prediction and a dashed one for the count.
    sx = np.array([s["x"] for s in sensors])
    sy = np.array([s["y"] for s in sensors])
    ax.plot(sx, sy, "o", ms=3.0, mfc=(*INK, 1), mec="none", zorder=8)
    peak = max(np.nanmax(s["predicted"]) for s in sensors)
    ring_pred = ax.scatter(
        sx,
        sy,
        s=1,
        facecolors="none",
        edgecolors=[(*INK, 0.95)],
        linewidths=1.1,
        zorder=8,
    )
    ring_obs = ax.scatter(
        sx,
        sy,
        s=1,
        facecolors="none",
        edgecolors=[(*INK, 0.8)],
        linewidths=0.9,
        linestyles=(0, (2, 2)),
        zorder=8,
    )
    top = sorted(sensors, key=lambda s: -max(v for v in s["predicted"]))[:12]
    labelled = []
    for s in top:
        if (
            len(mark_xy)
            and np.min(np.hypot(mark_xy[:, 0] - s["x"], mark_xy[:, 1] - s["y"])) < 140
        ):
            continue
        labelled.append(s)
    lab_txt = {}
    for s in labelled:
        lab_txt[s["location_id"]] = ax.text(
            s["x"] + 26,
            s["y"] - 40,
            "",
            fontname=FONT,
            fontsize=11.5,
            color=(*INK, 0.92),
            path_effects=[pe.withStroke(linewidth=2.6, foreground=(*PARCHMENT, 0.9))],
            zorder=9,
        )
    # Footprints.
    foot_w = EllipseCollection(
        widths=[1],
        heights=[1],
        angles=[0],
        units="xy",
        offsets=np.zeros((1, 2)),
        offset_transform=ax.transData,
        facecolors=[(*INK, 0)],
        edgecolors="none",
        zorder=4,
    )
    heel_w = EllipseCollection(
        widths=[1],
        heights=[1],
        angles=[0],
        units="xy",
        offsets=np.zeros((1, 2)),
        offset_transform=ax.transData,
        facecolors=[(*INK, 0)],
        edgecolors="none",
        zorder=4,
    )
    ax.add_collection(foot_w)
    ax.add_collection(heel_w)
    # Title, clock, timeline, legend.
    title = ax.text(
        0.5,
        0.5,
        "I solemnly swear that I am up to no good.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontname=FONT,
        fontsize=40,
        color=(*INK, 0),
        zorder=20,
    )
    clock = ax.text(
        VIEW[0] + 100,
        VIEW[3] - 150,
        "",
        fontname=FONT,
        fontsize=36,
        color=(*INK, 0.95),
        zorder=12,
        path_effects=[pe.withStroke(linewidth=4, foreground=(*PARCHMENT, 0.9))],
    )
    sub = ax.text(
        VIEW[0] + 100,
        VIEW[3] - 232,
        "Melbourne CBD  ·  predicted foot traffic  ·  Friday 6 November 2026",
        fontname=FONT,
        fontsize=15,
        color=(*INK, 0.85),
        zorder=12,
        path_effects=[pe.withStroke(linewidth=3, foreground=(*PARCHMENT, 0.9))],
    )
    legend = ax.text(
        VIEW[0] + 100,
        VIEW[1] + 215,
        "more footprints, more people\n●  sensor    ○  predicted    ┈  counted",
        fontname=FONT,
        fontsize=13.5,
        color=(*INK, 0.85),
        zorder=12,
        linespacing=1.5,
        path_effects=[pe.withStroke(linewidth=3, foreground=(*PARCHMENT, 0.9))],
    )
    foot = ax.text(
        (VIEW[0] + VIEW[2]) / 2,
        VIEW[1] + 95,
        "Stoa flow model  ·  planning guidance, not a crowd-safety assessment",
        fontname=FONT,
        fontsize=12,
        color=(*INK, 0.75),
        ha="center",
        zorder=12,
        path_effects=[pe.withStroke(linewidth=3, foreground=(*PARCHMENT, 0.9))],
    )
    end_title = ax.text(
        0.5,
        0.5,
        "Mischief managed.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontname=FONT,
        fontsize=52,
        color=(*INK, 0),
        zorder=20,
    )
    tl_x0, tl_x1 = VIEW[0] + 360, VIEW[2] - 360
    tl_y = VIEW[1] + 150
    ax.plot([tl_x0, tl_x1], [tl_y, tl_y], color=(*INK, 0.6), lw=0.8, zorder=12)
    for h in range(0, 25, 3):
        x = tl_x0 + (tl_x1 - tl_x0) * h / 24
        ax.plot([x, x], [tl_y - 9, tl_y + 9], color=(*INK, 0.7), lw=0.8, zorder=12)
        ax.text(
            x,
            tl_y - 52,
            f"{h % 24:02d}",
            ha="center",
            fontname=FONT,
            fontsize=11,
            color=(*INK, 0.75),
            zorder=12,
        )
    (cursor,) = ax.plot(
        [tl_x0], [tl_y], marker="v", ms=9, mfc=(*INK, 1), mec="none", zorder=13
    )

    walkers = Walkers(segs, np.nan_to_num(flow, nan=1.0))
    totals = np.nansum(flow, axis=0)
    spawn_max = 26.0
    mask_d = np.hypot(
        (np.arange(W)[None, :] / W - 0.5) * (VIEW[2] - VIEW[0]),
        (0.5 - np.arange(H)[:, None] / H) * (VIEW[3] - VIEW[1]),
    )
    rmax = float(mask_d.max())

    intro, outro = 70, 55
    n_main = 24 * args.hold
    total = intro + n_main + outro

    def frame(i: int) -> np.ndarray:
        # Phase and time of day.
        if i < intro:
            t = 0.0
            reveal = smooth((i - 28) / 38)
            ta = smooth(i / 20) * (1 - smooth((i - 48) / 18))
            map_a = 1.0
            end_a = 0.0
            hud = smooth((i - 50) / 20)
            spawn = 0.0
        elif i < intro + n_main:
            t = (i - intro) / args.hold
            reveal, ta, map_a, end_a, hud = 1.0, 0.0, 1.0, 0.0, 1.0
        else:
            t = 23.99
            k = i - intro - n_main
            reveal, ta = 1.0, 0.0
            map_a = 1 - 0.8 * smooth(k / 30)
            end_a = smooth((k - 14) / 30)
            hud = 1 - smooth(k / 22)
        if intro <= i < intro + n_main or i >= intro + n_main:
            hh = min(int(t), 23)
            share = (
                totals[hh] * (1 - (t - hh)) + totals[(hh + 1) % 24] * (t - hh)
            ) / totals.max()
            spawn = spawn_max * share**0.9 * (map_a if i >= intro + n_main else 1.0)
        walkers.step(t, spawn)
        # Ink reveal.
        a = ink.copy()
        if reveal < 1.0:
            edge = np.clip((reveal * rmax * 1.15 - mask_d) / 220.0, 0, 1)
            a[..., 3] *= edge
        a[..., 3] *= map_a
        ink_im.set_data(a)
        # Footprints.
        n = len(walkers.fx)
        if n:
            age = walkers.fage / walkers.LIFE
            alpha = 0.88 * (1 - age) ** 1.3 * map_a * reveal
            ang = walkers.fa
            off = np.c_[walkers.fx, walkers.fy]
            rad = np.radians(ang)
            foot_w.set_offsets(off + np.c_[np.cos(rad), np.sin(rad)] * 2.0)
            foot_w._widths = np.full(n, 8.5) / 2  # EllipseCollection stores half-axes
            foot_w._heights = np.full(n, 4.8) / 2
            foot_w._angles = np.radians(ang)
            heel_off = off - np.c_[np.cos(rad), np.sin(rad)] * 3.6
            heel_w.set_offsets(heel_off)
            heel_w._widths = np.full(n, 3.4) / 2
            heel_w._heights = np.full(n, 3.4) / 2
            heel_w._angles = np.radians(ang)
            col = np.c_[np.tile(INK, (n, 1)), alpha]
            foot_w.set_facecolor(col)
            heel_w.set_facecolor(col)
            foot_w.stale = heel_w.stale = True
        else:
            foot_w.set_facecolor([(*INK, 0)])
            heel_w.set_facecolor([(*INK, 0)])
        # Sensors: rings scale with the hour's prediction and count.
        hh = min(int(t), 23)
        f = t - hh
        h2 = (hh + 1) % 24
        sp = np.array(
            [(1 - f) * s["predicted"][hh] + f * s["predicted"][h2] for s in sensors]
        )
        so = np.array(
            [
                np.nan
                if s["observed"][hh] is None or s["observed"][h2] is None
                else (1 - f) * s["observed"][hh] + f * s["observed"][h2]
                for s in sensors
            ]
        )

        # Ring area is in points squared: radius ~ sqrt(count), capped by the peak.
        def area(v: np.ndarray) -> np.ndarray:
            r_pts = 3.0 + 17.0 * np.sqrt(np.nan_to_num(v) / peak)
            return np.pi * r_pts**2

        ring_pred.set_sizes(area(sp))
        ring_obs.set_sizes(area(so))
        sa = hud if i < intro else (hud if i >= intro + n_main else 1.0)
        ring_pred.set_alpha(sa * map_a)
        ring_obs.set_alpha(sa * map_a * 0.9)
        for s in labelled:
            j = sensors.index(s)
            txt = lab_txt[s["location_id"]]
            if np.isnan(so[j]):
                txt.set_text(f"{s['name']}\n{sp[j]:,.0f} predicted")
            else:
                txt.set_text(
                    f"{s['name']}\n{sp[j]:,.0f} predicted · {so[j]:,.0f} counted"
                )
            txt.set_alpha(sa * map_a)
        # HUD.
        hour = int(t) % 24
        minute = int((t - int(t)) * 60) // 5 * 5
        clock.set_text(f"{hour:02d}:{minute:02d}")
        for artist in (clock, sub, legend, foot):
            artist.set_alpha(hud)
        cursor.set_xdata([tl_x0 + (tl_x1 - tl_x0) * min(t, 24) / 24])
        cursor.set_alpha(hud)
        title.set_color((*INK, ta))
        end_title.set_color((*INK, end_a))
        for m_art in ax.texts:
            pass
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    if args.still is not None:
        # Warm up the walkers to the chosen hour, then save one frame.
        i0 = intro + int(args.still * args.hold)
        for i in range(max(intro, i0 - 150), i0):
            frame(i)
        img = frame(i0)
        out.mkdir(parents=True, exist_ok=True)
        plt.imsave(out / f"still_{int(args.still):02d}h.png", img)
        print("wrote", out / f"still_{int(args.still):02d}h.png")
        return

    import imageio.v2 as imageio
    from PIL import Image

    mp4 = out / "marauders_map.mp4"
    writer = imageio.get_writer(
        str(mp4), fps=args.fps, codec="libx264", quality=8, macro_block_size=2
    )
    gif_frames = []
    for i in range(total):
        img = frame(i)
        writer.append_data(img)
        if i % 2 == 0:
            gif_frames.append(
                Image.fromarray(img).resize((W // 2, H // 2), Image.LANCZOS)
            )
        if i % 60 == 0:
            print(f"frame {i}/{total}", flush=True)
    writer.close()
    pal = [
        f.quantize(colors=40, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for f in gif_frames
    ]
    pal[0].save(
        out / "marauders_map.gif",
        save_all=True,
        append_images=pal[1:],
        duration=int(2000 / args.fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    print("wrote", mp4, out / "marauders_map.gif")
    del meta


if __name__ == "__main__":
    main()
