"""Predictions for the Marauder's Map: the flow model on every CBD segment, hour by hour.

Usage: python -m src.simulation.marauders_map.predict [--date 2026-11-06]

Loads the registered `stoa-flow-rich` champion (v1: the mean model and its 10% and 90% quantile
models, from the MLflow artifact store; `--models` points at the downloaded .skops files),
builds the exact feature columns the model trained on for each segment and hour, and writes to
data/simulation/marauders_map/:
- segments.parquet: the CBD segments (segment_id, geometry in the site CRS);
- predictions.parquet: segment_id, hour, predicted, lower, upper (people per hour);
- sensors.parquet: each CBD sensor with its name, position, observed and predicted count per
  hour (the model trained on these sensors, so this is a consistency check, not a test);
- meta.json: the date, the model run and the assumptions below.

Assumptions (also in meta.json, because they are not what the model saw in training):
- Hour-level columns (weather, calendar, sun, COVID and Metro Tunnel flags) come from the
  hourly table's rows for that date, the same for every segment.
- `slope_pct`, `slope_pct_local_mean` and `elevation_m` exist only at sensors (the footpath
  steepness survey). At a sensor's own segment the survey values are used; elsewhere
  `slope_pct*` are the segment's DEM slope and `elevation_m` the nearest sensor's elevation.
- Event columns come from the event log and AFL fixtures when the raw crowd file is present.
  Without it only the logged events count.

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
from scipy.spatial import cKDTree

from src.features import shade as sh
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
MODELS = REPO_ROOT / "data" / "models" / "registered_rich"
RUN_ID = "2b730d3086a14685ba53467982389672"
# The registered champion's models (run `ci__20261116T005046`, data version 961075e6).
MEAN_MODEL = "m-d4172e239fe24d9e885eff49a5303504"
LOW_MODEL = "m-cc842e8d75ce425ebb8ea5047623a580"
HIGH_MODEL = "m-55d1c62f49aa46ee87546e0a442de5d5"
# The CBD and its edges (lon_min, lat_min, lon_max, lat_max).
BBOX_LONLAT = (144.944, -37.8305, 144.990, -37.8000)


def _load_model(name: str):
    import skops.io as sio

    path = MODELS / f"{name}.skops"
    return sio.load(path, trusted=sio.get_untrusted_types(file=path))


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
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data_cfg, fcfg = load_config(), load_features_config()
    site = site_settings(SITE, data_cfg, fcfg)
    feats_dir = REPO_ROOT / fcfg["processed_dir"] / "features" / SITE
    names = json.loads((MODELS / "features.json").read_text())

    row_cols = tf.row_features("rich")
    hourly = hour_rows(args.date, row_cols, feats_dir)
    per_hour = (
        hourly.sort_values("location_id")
        .groupby("hour_local")
        .first()
        .reindex(range(24))
    )
    per_hour["hour_local"] = per_hour.index
    if per_hour["ts_local"].isna().any():
        raise SystemExit(f"{args.date} lacks some hours in the hourly table")

    segs = cbd_segments(site)
    seg_cols = tf.segment_features("rich")
    osm = pd.read_parquet(feats_dir / "osm_segment_features.parquet").set_index(
        "segment_id"
    )
    osm = osm.loc[segs["segment_id"]]
    seg_mat = osm[[c for c in seg_cols if c in osm.columns]].astype("float32").copy()

    # Slope and elevation exist only at sensors: use the survey there, proxies elsewhere.
    steep = pd.read_parquet(feats_dir / "sensor_steepness.parquet")
    smap = pd.read_parquet(feats_dir / "sensor_osm_features.parquet")[
        ["location_id", "segment_id", "sensor_description", "location_type"]
    ]
    steep = steep.merge(smap[["location_id", "segment_id"]], on="location_id")
    mid = segs.geometry.interpolate(0.5, normalized=True)
    xy = np.c_[mid.x, mid.y]
    all_edges = gpd.read_parquet(
        site.graph_dir / "edges.parquet",
        columns=["segment_id", "geometry"],
    ).set_index("segment_id")
    s_mid = all_edges.geometry.loc[steep["segment_id"]].interpolate(
        0.5, normalized=True
    )
    tree = cKDTree(np.c_[s_mid.x, s_mid.y])
    near = tree.query(xy)[1]
    dem = osm["dem_slope_mean_pct"].astype("float32").to_numpy()
    seg_mat["slope_pct"] = dem
    seg_mat["slope_pct_local_mean"] = dem
    seg_mat["elevation_m"] = steep["elevation_m"].to_numpy(dtype="float32")[near]
    pos = pd.Series(np.arange(len(segs)), index=segs["segment_id"])
    for _, r in steep.dropna(subset=["segment_id"]).iterrows():
        i = pos.get(r["segment_id"])
        if i is not None:
            for c in ("slope_pct", "slope_pct_local_mean", "elevation_m"):
                if pd.notna(r[c]):
                    seg_mat.iloc[i, seg_mat.columns.get_loc(c)] = r[c]

    # Long frame: every segment x every hour.
    n_seg, n_h = len(segs), 24
    seg_rep = np.repeat(np.arange(n_seg), n_h)
    hour_rep = np.tile(np.arange(n_h), n_seg)
    frame = pd.DataFrame(index=np.arange(n_seg * n_h))
    for c in row_cols:
        if c in ("slope_pct", "slope_pct_local_mean", "elevation_m"):
            continue
        frame[c] = per_hour[c].to_numpy()[hour_rep].astype("float32")
    for c in ("slope_pct", "slope_pct_local_mean", "elevation_m"):
        frame[c] = seg_mat[c].to_numpy()[seg_rep]
    seg_ids = segs["segment_id"].to_numpy()[seg_rep]

    # Shade at the hour's sun position, and events near the segment.
    meta_sh = json.loads((feats_dir / "shade_meta.json").read_text())
    b = sh.bin_of(
        per_hour["sun_elevation_deg"].to_numpy()[hour_rep],
        per_hour["sun_azimuth_deg"].to_numpy()[hour_rep],
        meta_sh,
    )
    table = pd.read_parquet(feats_dir / "osm_segment_shade.parquet")
    table = table[table["segment_id"].isin(segs["segment_id"])]
    key = pd.MultiIndex.from_arrays([seg_ids, b])
    for name in ("shade_portable", "shade_rich"):
        look = table.set_index(["segment_id", "sun_bin"])[name]
        v = look.reindex(key).to_numpy(dtype="float32")
        v = np.where(np.isnan(v), 0.0, v)
        frame[name] = np.where(b < 0, 1.0, v).astype("float32")
    ts = pd.Series(per_hour["ts_local"].to_numpy()[hour_rep])
    ev = tf.event_columns(ts, pd.Series(seg_ids), SITE, data_cfg, fcfg)
    for c in ev.columns:
        frame[c] = ev[c].to_numpy().astype("float32")

    seg_part = seg_mat.drop(
        columns=["slope_pct", "slope_pct_local_mean", "elevation_m"]
    )
    if "recent_station_dist_m" in seg_part.columns and not bool(
        per_hour["post_metro_tunnel"].iloc[0]
    ):
        seg_part["recent_station_dist_m"] = fcfg["built_form"]["max_distance_m"]
    seg_block = seg_part.to_numpy("float32")[seg_rep]
    seg_df = pd.DataFrame(seg_block, columns=seg_part.columns, index=frame.index)
    full = pd.concat([frame, seg_df], axis=1)
    missing = [c for c in names if c not in full.columns]
    if missing:
        raise SystemExit(f"missing model features: {missing[:8]}")
    x = full[names].to_numpy("float32")

    from src.models.flow import bundle, candidates

    mean = candidates.predict(_load_model(MEAN_MODEL), x)
    low = candidates.predict(_load_model(LOW_MODEL), x)
    high = candidates.predict(_load_model(HIGH_MODEL), x)
    # The registered v1 margins are not stored beside the mean model; the quantiles are
    # shown unwidened, and kept around the mean.
    low, high = np.minimum(low, mean), np.maximum(high, mean)
    pred = pd.DataFrame(
        {
            "segment_id": seg_ids,
            "hour": hour_rep.astype("int8"),
            "predicted": mean.astype("float32"),
            "lower": low.astype("float32"),
            "upper": high.astype("float32"),
        }
    )
    del bundle

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
        "model": "stoa-flow-rich v1 (champion), LightGBM Poisson mean",
        "run_id": RUN_ID,
        "segments": int(n_seg),
        "sensors": len(sensors),
        "sensor_hours_compared": len(o),
        "sensor_wape": float(np.abs(o - p).sum() / max(o.sum(), 1.0)),
        "sensor_log_mae": float(np.abs(np.log1p(o) - np.log1p(p)).mean()),
        "assumptions": [
            "hour-level columns from the hourly table's rows for the date",
            "slope_pct* = DEM slope and elevation_m = nearest sensor off-sensor",
            "quantile bounds are the registered models' raw 10%/90%, unwidened",
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
