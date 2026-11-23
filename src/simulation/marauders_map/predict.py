"""Predictions for the Marauder's Map: the flow model on every CBD segment, hour by hour.

Usage: python -m src.simulation.marauders_map.predict [--date 2026-11-06]

Loads the registered `stoa-flow-rich` champion from MLflow (`--model`, default
models:/stoa-flow-rich@champion; on the VM the pipeline's MLFLOW_TRACKING_URI points at the
team's server), builds every segment's features for each hour of `--date` with the
prediction API's code (src/api/features.py, the path tests/api/test_parity.py checks
against training), and writes to data/simulation/marauders_map/:
- segments.parquet: the CBD segments (segment_id, geometry in the site CRS);
- predictions.parquet: segment_id, hour, predicted, lower, upper (people per hour; the
  bundle's 80% interval);
- sensors.json: each CBD sensor with its name, position, observed and predicted count per
  hour (the model trained on these sensors, so this is a consistency check, not a test);
- meta.json: the date, the model version and run, and how the inputs were made.

The date is a replay of a past day, so its weather is the observed archive training uses
(counts_hourly.read_weather), not a forecast. Slope and elevation come from
segment_steepness (a sensor's own values where one sits on the segment, the footpath survey
or the DEM elsewhere).

Numbers shown in the animation come from this file only (CLAUDE.md, core rule 1).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from src.api import model as model_mod
from src.api.features import SiteTables, features_for_rows, grid, localize
from src.features import counts_hourly as ch
from src.features import training_frame as tf
from src.features.common import load_features_config, site_settings
from src.ingest.common import REPO_ROOT, load_config, setup_logging


def _default_site() -> str:
    sites = load_config()["sites"]
    return next(
        (name for name, cfg in sites.items() if cfg.get("count_sensors")),
        next(iter(sites)),
    )


SITE = _default_site()
OUT = REPO_ROOT / "data" / "simulation" / "marauders_map"
MODEL_URI = "models:/stoa-flow-rich@champion"
# The CBD and its edges (lon_min, lat_min, lon_max, lat_max).
BBOX_LONLAT = (144.944, -37.8305, 144.990, -37.8000)


def cbd_segments(site_cfg) -> gpd.GeoDataFrame:
    """Segments whose midpoint lies inside the CBD box."""
    edges = gpd.read_parquet(
        site_cfg.graph_dir / "edges.parquet",
        columns=["segment_id", "highway", "geometry"],
    )
    t = Transformer.from_crs("EPSG:4326", site_cfg.crs, always_xy=True)
    x0, y0 = t.transform(BBOX_LONLAT[0], BBOX_LONLAT[1])
    x1, y1 = t.transform(BBOX_LONLAT[2], BBOX_LONLAT[3])
    mid = edges.geometry.interpolate(0.5, normalized=True)
    keep = (mid.x >= x0) & (mid.x <= x1) & (mid.y >= y0) & (mid.y <= y1)
    return edges[keep].reset_index(drop=True)


def hour_rows(date: str, row_cols: list[str], feats_dir: Path) -> pd.DataFrame:
    """One row per hour of `date` with the hour-level columns, from the hourly table."""
    feats = feats_dir / "counts_hourly.parquet"
    keep = list(
        dict.fromkeys(
            ["location_id", "ts_local", "date_local", "pedestriancount"]
            + ["sun_azimuth_deg", "suspect_outage_zero"]
            + row_cols
        )
    )
    h = pd.read_parquet(feats, columns=keep)
    h = h[h["date_local"] == pd.Timestamp(date)]
    if h.empty:
        raise SystemExit(f"no hourly rows for {date}")
    return h


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-11-06")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--model", default=MODEL_URI, help="any MLflow model URI or folder")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data_cfg, fcfg = load_config(), load_features_config()
    site = site_settings(SITE, data_cfg, fcfg)
    feats_dir = REPO_ROOT / fcfg["processed_dir"] / "features" / SITE
    model = model_mod.load("rich", args.model)

    row_cols = tf.row_features("rich")
    hourly = hour_rows(args.date, row_cols, feats_dir)

    # Every CBD segment at every hour of the date, built as the API builds a request.
    segs = cbd_segments(site)
    tables = SiteTables.load(SITE)
    times = localize(
        list(pd.date_range(args.date, periods=24, freq="h")), tables.timezone
    )
    if len(times) != 24:
        raise SystemExit(f"{args.date} doesn't have 24 clock hours (daylight saving)")
    rows = grid(list(segs["segment_id"]), times)
    weather = ch.read_weather(REPO_ROOT / data_cfg["raw_dir"] / "weather", SITE)
    x = features_for_rows(tables, rows, weather, "rich")
    wx_cols = [c for c in x.columns if c.startswith("wx_")]
    if x[wx_cols].isna().any(axis=None):
        raise SystemExit(f"the weather archive doesn't cover all of {args.date}")
    out_pred = model.predict(x)
    n_seg = len(segs)
    seg_ids = rows["segment_id"].to_numpy()
    hour_rep = rows["ts_local"].dt.hour.to_numpy()
    pred = pd.DataFrame(
        {
            "segment_id": seg_ids,
            "hour": hour_rep.astype("int8"),
            "predicted": out_pred["predicted"].to_numpy(dtype="float32"),
            "lower": out_pred["lower"].to_numpy(dtype="float32"),
            "upper": out_pred["upper"].to_numpy(dtype="float32"),
        }
    )
    smap = pd.read_parquet(feats_dir / "sensor_osm_features.parquet")[
        ["location_id", "segment_id", "sensor_description", "location_type"]
    ]
    mid = segs.geometry.interpolate(0.5, normalized=True)
    xy = np.c_[mid.x, mid.y]
    pos = pd.Series(np.arange(len(segs)), index=segs["segment_id"])

    # Sensors in the box: observed and predicted counts per hour.
    inside = set(segs["segment_id"])
    s = smap[smap["segment_id"].isin(inside)]
    s = s[s["location_type"].eq("Outdoor")]
    obs = hourly[hourly["location_id"].isin(s["location_id"])]
    obs = obs[~obs["suspect_outage_zero"].fillna(False).astype(bool)]
    obs = obs.pivot_table(
        index="location_id", columns="hour_local", values="pedestriancount"
    )
    pp = pred.pivot_table(index="segment_id", columns="hour", values="predicted")
    rows = []
    for _, r in s.iterrows():
        i = pos[r["segment_id"]]
        rows.append(
            {
                "location_id": int(r["location_id"]),
                "name": r["sensor_description"],
                "segment_id": r["segment_id"],
                "x": float(xy[i, 0]),
                "y": float(xy[i, 1]),
                "observed": [
                    None
                    if pd.isna(obs.get(h, pd.Series()).get(r["location_id"]))
                    else float(obs[h][r["location_id"]])
                    for h in range(24)
                ],
                "predicted": [float(pp.loc[r["segment_id"], h]) for h in range(24)],
            }
        )
    sensors = pd.DataFrame(rows)

    segs.to_parquet(out / "segments.parquet")
    pred.to_parquet(out / "predictions.parquet", index=False)
    sensors.to_json(out / "sensors.json", orient="records", indent=1)
    both = (
        sensors.apply(
            lambda r: [
                (o, p) for o, p in zip(r["observed"], r["predicted"]) if o is not None
            ],
            axis=1,
        )
        .explode()
        .dropna()
    )
    o = np.array([v[0] for v in both])
    p = np.array([v[1] for v in both])
    meta = {
        "date": args.date,
        "model": f"{model.label} ({args.model})",
        "run_id": model.run_id or "unknown",
        "segments": int(n_seg),
        "sensors": len(sensors),
        "sensor_hours_compared": len(o),
        "sensor_wape": float(np.abs(o - p).sum() / max(o.sum(), 1.0)),
        "sensor_log_mae": float(np.abs(np.log1p(o) - np.log1p(p)).mean()),
        "assumptions": [
            "features built by src/api/features.py, the path checked against training",
            "weather: the observed archive for the date (a replay, not a forecast)",
            "slope and elevation from segment_steepness (sensor, footpath survey or DEM)",
            "bounds: the registered bundle's 80% interval",
        ],
        "extent_bbox_lonlat": BBOX_LONLAT,
        "crs": site.crs,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(pred.groupby("hour")["predicted"].sum().round(0).to_string())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
