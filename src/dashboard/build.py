"""Build the universal STOA purple & orange Liquid Glass monitoring, real-map, and simulation dashboard.

Usage:
    python -m src.dashboard.build [--sim-root data/simulation] [--out data/dashboard]

Works for any new model predictions, scenarios, or city views:
- Auto-discovers every scenario/city directory under ``--sim-root`` containing
  ``meta.json`` and ``sensors.json`` (plus optional ``landmarks.json``, ``*.mp4``,
  ``counts_hourly_meta.json``, and ``slice_errors.csv``).
- Projects coordinates from any CRS to WGS84 (lat/lon) using ``pyproj`` when installed,
  or a built-in universal UTM / WGS84 / extent-bbox fallback using only the Python
  standard library.
- Dynamically computes map bounds and center for each city/scenario so Leaflet flies
  to Melbourne, NEU Boston, NYC, or any custom site automatically.
- Handles both observed+predicted monitoring scenarios and forecast-only scenarios
  (such as NEU Boston or future what-if scenarios without ground-truth counts).
- Strictly grounded in model outputs and real sensor counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_ROOT = REPO_ROOT / "data" / "simulation"
DEFAULT_VIDEO = REPO_ROOT / "data" / "simulation" / "marauders_chronicle" / "marauders_chronicle.mp4"
DEFAULT_COUNTS_META = REPO_ROOT / "data" / "processed" / "features" / "melbourne" / "counts_hourly_meta.json"
DEFAULT_BIAS_CSV = REPO_ROOT / "data" / "processed" / "bias" / "melbourne" / "slice_errors.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "dashboard"

# Fallback Melbourne CBD landmarks (used only when a scenario is in Melbourne and lacks landmarks.json)
MELBOURNE_LANDMARKS_UTM: list[dict[str, Any]] = [
    {"name": "Flinders Street Station", "category": "transit", "x": 321020.0, "y": 5812445.0},
    {"name": "Southern Cross Station", "category": "transit", "x": 319820.0, "y": 5812455.0},
    {"name": "Melbourne Central", "category": "transit", "x": 320815.0, "y": 5813215.0},
    {"name": "Parliament Station", "category": "transit", "x": 321700.0, "y": 5813160.0},
    {"name": "Bourke Street Mall", "category": "civic", "x": 320885.0, "y": 5812908.0},
    {"name": "Queen Victoria Market", "category": "civic", "x": 320320.0, "y": 5813665.0},
    {"name": "Town Hall", "category": "civic", "x": 320985.0, "y": 5812780.0},
    {"name": "Flagstaff Gardens", "category": "park", "x": 320045.0, "y": 5813325.0},
    {"name": "Treasury Gardens", "category": "park", "x": 321895.0, "y": 5812655.0},
    {"name": "Birrarung Marr", "category": "park", "x": 321520.0, "y": 5812370.0},
    {"name": "Southbank", "category": "park", "x": 320890.0, "y": 5812145.0},
]


def utm_to_latlon(easting: float, northing: float, zone: int, northern: bool) -> tuple[float, float]:
    """Convert UTM / MGA projected coordinates to WGS84 (lat, lon) in pure Python."""
    a = 6378137.0
    f = 1.0 / 298.257222101
    e2 = f * (2.0 - f)
    e_prime2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    x = easting - 500000.0
    y = northing if northern else (northing - 10000000.0)

    m = y / k0
    mu = m / (a * (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0))
    e1 = (1.0 - math.sqrt(1.0 - e2)) / (1.0 + math.sqrt(1.0 - e2))

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_p = math.sin(phi1)
    cos_p = math.cos(phi1)
    tan_p = math.tan(phi1)
    n1 = a / math.sqrt(1.0 - e2 * sin_p**2)
    t1 = tan_p**2
    c1 = e_prime2 * cos_p**2
    r1 = a * (1.0 - e2) / ((1.0 - e2 * sin_p**2) ** 1.5)
    d = x / (n1 * k0)

    lat = phi1 - (n1 * tan_p / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * e_prime2) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2 - 252.0 * e_prime2 - 3.0 * c1**2)
        * d**6
        / 720.0
    )
    lon = lon0 + (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * e_prime2 + 24.0 * t1**2) * d**5 / 120.0
    ) / cos_p

    return round(math.degrees(lat), 6), round(math.degrees(lon), 6)


def _parse_epsg_code(crs: Any) -> int | None:
    if crs is None:
        return None
    if isinstance(crs, int):
        return crs
    m = re.search(r"(\d{4,5})", str(crs))
    return int(m.group(1)) if m else None


def make_projector(
    crs: Any,
    xy_bounds: tuple[float, float, float, float] | None = None,
    bbox_lonlat: list[float] | tuple[float, ...] | None = None,
):
    """Return a callable (x, y) -> (lat, lon) supporting any CRS (pyproj or stdlib fallback)."""
    epsg = _parse_epsg_code(crs)
    if epsg == 4326:
        return lambda x, y: (round(float(y), 6), round(float(x), 6))

    # Prefer pyproj when available (handles every CRS on Earth, e.g. EPSG:26986, 2263, 28355).
    if crs is not None:
        try:
            from pyproj import Transformer

            code = f"EPSG:{epsg}" if epsg is not None else str(crs)
            tr = Transformer.from_crs(code, "EPSG:4326", always_xy=True)
            return lambda x, y: (
                round(float(tr.transform(float(x), float(y))[1]), 6),
                round(float(tr.transform(float(x), float(y))[0]), 6),
            )
        except Exception:
            pass

    # Pure-stdlib universal UTM / MGA / NAD83-UTM fallback
    if epsg is not None:
        if 32601 <= epsg <= 32660:
            zone = epsg - 32600
            return lambda x, y: utm_to_latlon(float(x), float(y), zone, northern=True)
        if 32701 <= epsg <= 32760:
            zone = epsg - 32700
            return lambda x, y: utm_to_latlon(float(x), float(y), zone, northern=False)
        if 28348 <= epsg <= 28358:
            zone = epsg - 28300
            return lambda x, y: utm_to_latlon(float(x), float(y), zone, northern=False)
        if 26901 <= epsg <= 26923:
            zone = epsg - 26900
            return lambda x, y: utm_to_latlon(float(x), float(y), zone, northern=True)

    # Last-resort linear bbox mapping using meta["extent_bbox_lonlat"]
    if xy_bounds and bbox_lonlat and len(bbox_lonlat) == 4:
        x0, y0, x1, y1 = xy_bounds
        lon0, lat0, lon1, lat1 = [float(v) for v in bbox_lonlat]
        dx = max(x1 - x0, 1e-6)
        dy = max(y1 - y0, 1e-6)
        return lambda x, y: (
            round(lat0 + (float(y) - y0) / dy * (lat1 - lat0), 6),
            round(lon0 + (float(x) - x0) / dx * (lon1 - lon0), 6),
        )

    # Default passthrough if coordinates are already geographic
    return lambda x, y: (round(float(y), 6), round(float(x), 6))


def short_sensor_name(name: str, max_len: int = 34) -> str:
    """Shorten verbose sensor descriptions for dashboard tables and popups."""
    cleaned = re.sub(r"^(I-Hub|iHub|COM Pole \d+\s*-\s*)\s*", "", name, flags=re.I)
    cleaned = re.sub(r"\s*-\s*New footpath$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\(COM Pole \d+\)$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def deduplicate_sensors_geo(
    sensors: list[dict[str, Any]],
    min_dist_deg: float = 0.00022,
) -> list[dict[str, Any]]:
    """Deduplicate co-located sensor poles (~25 m threshold in WGS84 degrees)."""

    def priority(s: dict[str, Any]) -> tuple[int, float]:
        obs_count = sum(1 for v in s.get("observed", []) if v is not None)
        peak_pred = max((v for v in s.get("predicted", []) if v is not None), default=0.0)
        return (1 if obs_count > 0 else 0, peak_pred)

    ordered = sorted(sensors, key=priority, reverse=True)
    kept: list[dict[str, Any]] = []
    for s in ordered:
        if any(
            math.hypot(s["lat"] - k["lat"], s["lon"] - k["lon"]) < min_dist_deg for k in kept
        ):
            continue
        kept.append(s)
    return kept


def classify_sensor_status(missing_hours: int, wape: float | None, has_any_obs: bool) -> str:
    """Classify sensor operational & model-fit status for monitoring alerts."""
    if not has_any_obs:
        return "Forecast Only"
    if missing_hours >= 24:
        return "Offline"
    if missing_hours >= 4:
        return "Partial Outage"
    if wape is not None and wape >= 0.40:
        return "High Drift"
    return "Healthy"


def _infer_site_label(scenario_id: str, meta: dict[str, Any], center: tuple[float, float]) -> str:
    if meta.get("site_label"):
        return str(meta["site_label"])
    if meta.get("scenario_name"):
        return str(meta["scenario_name"])
    lat, lon = center
    if -38.1 <= lat <= -37.5 and 144.7 <= lon <= 145.3:
        return "Melbourne CBD"
    if 42.1 <= lat <= 42.5 and -71.3 <= lon <= -70.9:
        return "NEU Boston"
    if 40.5 <= lat <= 40.95 and -74.25 <= lon <= -73.7:
        return "New York City"
    return scenario_id.replace("_", " ").replace("-", " ").title()


def compute_scenario_payload(
    scenario_id: str,
    meta: dict[str, Any],
    raw_sensors: list[dict[str, Any]],
    raw_landmarks: list[dict[str, Any]] | None = None,
    counts_meta: dict[str, Any] | None = None,
    bias_rows: list[dict[str, Any]] | None = None,
    video_relpath: str | None = None,
) -> dict[str, Any]:
    """Build a self-contained monitoring & real-map payload for any city/scenario."""
    xs = [float(s["x"]) for s in raw_sensors if "x" in s and s["x"] is not None]
    ys = [float(s["y"]) for s in raw_sensors if "y" in s and s["y"] is not None]
    xy_bounds = (min(xs), min(ys), max(xs), max(ys)) if xs and ys else None
    project = make_projector(meta.get("crs"), xy_bounds, meta.get("extent_bbox_lonlat"))

    # Attach WGS84 lat/lon to every sensor
    projected_sensors: list[dict[str, Any]] = []
    for s in raw_sensors:
        if "lat" in s and "lon" in s and s["lat"] is not None and s["lon"] is not None:
            lat, lon = round(float(s["lat"]), 6), round(float(s["lon"]), 6)
        else:
            lat, lon = project(float(s["x"]), float(s["y"]))
        projected_sensors.append({**s, "lat": lat, "lon": lon})

    deduped = deduplicate_sensors_geo(projected_sensors)
    has_any_obs = any(
        v is not None for s in raw_sensors for v in (s.get("observed") or [])
    )

    # Hourly aggregates across all sensors
    hourly: list[dict[str, Any]] = []
    for h in range(24):
        pred_all = [
            float(s["predicted"][h])
            for s in raw_sensors
            if h < len(s.get("predicted", [])) and s["predicted"][h] is not None
        ]
        pairs = [
            (float(s["predicted"][h]), float(s["observed"][h]))
            for s in raw_sensors
            if h < len(s.get("predicted", []))
            and h < len(s.get("observed", []))
            and s["predicted"][h] is not None
            and s["observed"][h] is not None
        ]
        sum_pred_all = sum(pred_all)
        peak_sensor = max(pred_all) if pred_all else 0.0
        mean_sensor = sum_pred_all / len(pred_all) if pred_all else 0.0

        sum_p = sum(p for p, _ in pairs)
        sum_o = sum(o for _, o in pairs)
        abs_err = sum(abs(p - o) for p, o in pairs)
        wape_h = (abs_err / sum_o) if sum_o > 0 else None
        mae_h = (abs_err / len(pairs)) if pairs else None
        bias_h = ((sum_p - sum_o) / sum_o) if sum_o > 0 else None
        hourly.append(
            {
                "hour": h,
                "sum_predicted_all": round(sum_pred_all, 1),
                "sum_predicted_paired": round(sum_p, 1),
                "sum_observed_paired": round(sum_o, 1),
                "peak_sensor": round(peak_sensor, 1),
                "mean_sensor": round(mean_sensor, 1),
                "reporting_sensors": len(pairs),
                "total_sensors": len(raw_sensors),
                "wape": round(wape_h, 4) if wape_h is not None else None,
                "mae": round(mae_h, 1) if mae_h is not None else None,
                "bias": round(bias_h, 4) if bias_h is not None else None,
            }
        )

    peak_hour_row = max(hourly, key=lambda r: r["sum_predicted_all"])
    max_total = max(peak_hour_row["sum_predicted_all"], 1.0)
    for row in hourly:
        row["activity_pct"] = round(100.0 * row["sum_predicted_all"] / max_total, 1)

    sensor_records: list[dict[str, Any]] = []
    status_counts = {
        "Healthy": 0,
        "High Drift": 0,
        "Partial Outage": 0,
        "Offline": 0,
        "Forecast Only": 0,
    }
    for s in deduped:
        preds = [round(float(v), 1) for v in s.get("predicted", [])]
        raw_obs = s.get("observed") or [None] * len(preds)
        obs = [None if v is None else round(float(v), 1) for v in raw_obs]
        pairs = [(p, o) for p, o in zip(preds, obs) if o is not None]
        missing = len(preds) - len(pairs)
        sum_p = sum(p for p, _ in pairs)
        sum_o = sum(o for _, o in pairs)
        abs_err = sum(abs(p - o) for p, o in pairs)
        wape_s = (abs_err / sum_o) if sum_o > 0 else None
        bias_s = ((sum_p - sum_o) / sum_o) if sum_o > 0 else None
        peak_val = max(preds) if preds else 0.0
        peak_h = preds.index(peak_val) if preds else 0
        status = classify_sensor_status(missing, wape_s, has_any_obs)
        status_counts[status] += 1

        sensor_records.append(
            {
                "id": int(s.get("location_id", len(sensor_records) + 1)),
                "name": short_sensor_name(str(s.get("name", f"Point {len(sensor_records) + 1}"))),
                "full_name": str(s.get("name", f"Point {len(sensor_records) + 1}")),
                "lat": s["lat"],
                "lon": s["lon"],
                "predicted": preds,
                "observed": obs,
                "daily_pred": round(sum(preds), 0),
                "daily_obs": round(sum_o, 0) if pairs else None,
                "peak_pred": round(peak_val, 0),
                "peak_hour": peak_h,
                "missing_hours": missing,
                "wape": round(wape_s, 4) if wape_s is not None else None,
                "bias": round(bias_s, 4) if bias_s is not None else None,
                "status": status,
            }
        )

    # Dynamic map bounds & center (works for any city on Earth!)
    lats = [s["lat"] for s in sensor_records]
    lons = [s["lon"] for s in sensor_records]
    bbox = meta.get("extent_bbox_lonlat")
    if lats and lons:
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
    elif bbox and len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    else:
        min_lat, max_lat, min_lon, max_lon = -37.825, -37.805, 144.950, 144.978

    center = (round((min_lat + max_lat) / 2.0, 6), round((min_lon + max_lon) / 2.0, 6))
    bounds = [[round(min_lat, 6), round(min_lon, 6)], [round(max_lat, 6), round(max_lon, 6)]]
    site_label = _infer_site_label(scenario_id, meta, center)

    # Dynamic landmarks (from scenario landmarks.json, or Melbourne fallback only for Melbourne)
    landmarks: list[dict[str, Any]] = []
    if raw_landmarks is not None:
        for lm in raw_landmarks:
            if "lat" in lm and "lon" in lm:
                llat, llon = round(float(lm["lat"]), 6), round(float(lm["lon"]), 6)
            else:
                llat, llon = project(float(lm["x"]), float(lm["y"]))
            landmarks.append(
                {
                    "name": str(lm["name"]),
                    "category": str(lm.get("category", "civic")),
                    "lat": llat,
                    "lon": llon,
                }
            )
    elif site_label == "Melbourne CBD":
        for lm in MELBOURNE_LANDMARKS_UTM:
            llat, llon = utm_to_latlon(lm["x"], lm["y"], zone=55, northern=False)
            landmarks.append(
                {"name": lm["name"], "category": lm["category"], "lat": llat, "lon": llon}
            )

    keep_dims = {
        "precinct",
        "time_of_day",
        "volume_tier",
        "rain",
        "temperature",
        "era",
        "holdout_precinct",
    }
    clean_bias: list[dict[str, Any]] = []
    for row in bias_rows or []:
        if row.get("dimension") not in keep_dims:
            continue
        if row.get("mitigation") not in {"none", "calibrated"}:
            continue
        clean_bias.append(
            {
                "mitigation": row["mitigation"],
                "dimension": row["dimension"],
                "group": row["group"],
                "wape": round(float(row["wape"]), 4),
                "bias": round(float(row["bias"]), 4),
                "mae": round(float(row["mae"]), 1),
                "n_sensors": int(float(row["n_sensors"])),
                "worse": str(row.get("worse", "")).lower() == "true",
                "disparity_ratio": round(float(row["disparity_ratio"]), 4),
            }
        )

    sensor_wape = meta.get("sensor_wape")
    sensor_log_mae = meta.get("sensor_log_mae")

    return {
        "id": scenario_id,
        "label": f"{site_label} · {meta.get('date', 'scenario')}",
        "site_label": site_label,
        "has_observed": has_any_obs,
        "has_video": video_relpath is not None,
        "video_file": video_relpath,
        "map_center": list(center),
        "map_bounds": bounds,
        "meta": {
            "date": meta.get("date", "unknown"),
            "model": meta.get("model", "STOA flow model"),
            "run_id": str(meta.get("run_id", "local")),
            "crs": str(meta.get("crs", "EPSG:4326")),
            "segments": int(meta.get("segments", 0)),
            "sensors_total": int(meta.get("sensors", len(raw_sensors))),
            "sensors_in_view": len(sensor_records),
            "sensor_hours_compared": int(meta.get("sensor_hours_compared", 0)),
            "sensor_wape": round(float(sensor_wape), 4) if sensor_wape is not None else None,
            "sensor_log_mae": round(float(sensor_log_mae), 4)
            if sensor_log_mae is not None
            else None,
            "peak_hour": peak_hour_row["hour"],
            "peak_sensor_flow": peak_hour_row["peak_sensor"],
            "assumptions": meta.get("assumptions", []),
        },
        "pipeline_health": {
            "rows": int(counts_meta.get("rows", 0)) if counts_meta else None,
            "sensors": int(counts_meta.get("sensors", 0)) if counts_meta else None,
            "first_hour_local": counts_meta.get("first_hour_local") if counts_meta else None,
            "last_hour_local": counts_meta.get("last_hour_local") if counts_meta else None,
            "rows_suspect_outage_zero": int(counts_meta.get("rows_suspect_outage_zero", 0))
            if counts_meta
            else None,
            "rows_without_weather": int(counts_meta.get("rows_without_weather", 0))
            if counts_meta
            else None,
            "rows_by_source": counts_meta.get("rows_by_source", {}) if counts_meta else {},
            "runtime_s": counts_meta.get("runtime_s") if counts_meta else None,
        },
        "status_counts": status_counts,
        "hourly": hourly,
        "sensors": sensor_records,
        "landmarks": landmarks,
        "bias_slices": clean_bias,
    }


def load_bias_csv(path: Path) -> list[dict[str, Any]]:
    """Read bias slice_errors.csv if present."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def discover_scenario_dirs(sim_root: Path) -> list[Path]:
    """Find every scenario/city directory under `sim_root` with meta.json + sensors.json."""
    if not sim_root.exists():
        return []
    found: list[Path] = []
    if (sim_root / "meta.json").exists() and (sim_root / "sensors.json").exists():
        found.append(sim_root)
    for child in sorted(sim_root.iterdir()):
        if child.is_dir() and (child / "meta.json").exists() and (child / "sensors.json").exists():
            found.append(child)
    return found


def render_html(bundle: dict[str, Any]) -> str:
    """Render the multi-scenario, multi-city purple & orange Apple Liquid Glass STOA dashboard."""
    data_json = json.dumps(bundle, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>STOA · Pedestrian Flow Monitoring, Real Map &amp; Simulation</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <style>
    :root {{
      --ink: #1E1338;
      --ink-soft: #5E4E7A;
      --blue: #7C3AED;
      --sky: #F97316;
      --cyan: #EA580C;
      --indigo: #9333EA;
      --emerald: #059669;
      --rose: #E11D48;
      --amber: #D97706;
      --glass-bg: linear-gradient(145deg, rgba(255, 255, 255, 0.74) 0%, rgba(248, 240, 255, 0.46) 56%, rgba(255, 243, 232, 0.46) 100%);
      --glass-bg-strong: linear-gradient(145deg, rgba(255, 255, 255, 0.88) 0%, rgba(250, 243, 255, 0.66) 100%);
      --glass-border: rgba(255, 255, 255, 0.84);
      --glass-rim: inset 0 1px 0 rgba(255, 255, 255, 0.96), inset 0 -1px 0 rgba(124, 58, 237, 0.12);
      --glass-shadow: 0 22px 54px rgba(49, 20, 96, 0.11), 0 2px 10px rgba(49, 20, 96, 0.05);
      --hairline: rgba(124, 58, 237, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Roboto, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(60% 52% at 8% 6%, rgba(196, 181, 253, 0.58) 0%, rgba(196, 181, 253, 0) 70%),
        radial-gradient(54% 48% at 90% 10%, rgba(253, 186, 116, 0.54) 0%, rgba(253, 186, 116, 0) 72%),
        radial-gradient(52% 52% at 82% 88%, rgba(216, 180, 254, 0.48) 0%, rgba(216, 180, 254, 0) 72%),
        radial-gradient(55% 48% at 14% 84%, rgba(254, 215, 170, 0.52) 0%, rgba(254, 215, 170, 0) 72%),
        #FBF7FF;
      background-attachment: fixed;
      -webkit-font-smoothing: antialiased;
      line-height: 1.5;
    }}
    .wrap {{
      max-width: 1460px; margin: 0 auto; padding: 30px 28px 64px;
      display: flex; flex-direction: column; gap: 26px;
    }}
    .glass {{
      background: var(--glass-bg);
      backdrop-filter: blur(28px) saturate(190%);
      -webkit-backdrop-filter: blur(28px) saturate(190%);
      border: 1px solid var(--glass-border);
      border-radius: 26px;
      box-shadow: var(--glass-shadow), var(--glass-rim);
    }}
    .glass-pill {{
      background: var(--glass-bg-strong);
      backdrop-filter: blur(20px) saturate(180%);
      -webkit-backdrop-filter: blur(20px) saturate(180%);
      border: 1px solid rgba(255, 255, 255, 0.88);
      border-radius: 999px;
      box-shadow: 0 4px 16px rgba(49, 20, 96, 0.07), inset 0 1px 0 rgba(255, 255, 255, 0.96);
    }}
    header.glass {{
      padding: 22px 28px; display: flex; flex-wrap: wrap; align-items: center;
      justify-content: space-between; gap: 18px;
    }}
    .brand-row {{ display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }}
    .brand-title {{
      font-size: 32px; font-weight: 800; letter-spacing: 0.10em; margin: 0; line-height: 1;
      background: linear-gradient(135deg, #2E1065 0%, #7C3AED 54%, #F97316 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .brand-tag {{ font-size: 13px; font-weight: 600; color: var(--blue); letter-spacing: 0.03em; }}
    .brand-sub {{ font-size: 13px; color: var(--ink-soft); margin-top: 6px; }}
    .pills {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .pill {{ padding: 7px 14px; font-size: 12.5px; font-weight: 500; display: inline-flex; align-items: center; gap: 8px; }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}

    .kpi-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 16px; }}
    @media (max-width: 1240px) {{ .kpi-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}
    @media (max-width: 680px) {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    .kpi {{ padding: 18px 20px; display: flex; flex-direction: column; gap: 5px; transition: transform 0.22s ease, box-shadow 0.22s ease; }}
    .kpi:hover {{ transform: translateY(-2px); box-shadow: 0 26px 56px rgba(124, 58, 237, 0.14), var(--glass-rim); }}
    .kpi-head {{ display: flex; align-items: center; justify-content: space-between; font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-soft); }}
    .kpi-val {{ font-size: 25px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; font-variant-numeric: tabular-nums; }}
    .kpi-sub {{ font-size: 11.8px; color: var(--ink-soft); }}

    .stage-panel {{ padding: 26px 28px; display: flex; flex-direction: column; gap: 20px; }}
    .section-head {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 14px; }}
    .section-title {{ font-size: 12.5px; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; color: var(--ink-soft); }}
    .segmented {{ display: inline-flex; flex-wrap: wrap; gap: 5px; padding: 5px; border-radius: 999px; background: rgba(124, 58, 237, 0.09); border: 1px solid rgba(255, 255, 255, 0.78); }}
    .seg-btn {{ font-family: inherit; font-size: 12.5px; font-weight: 600; border: none; background: transparent; color: var(--ink-soft); padding: 7px 16px; border-radius: 999px; cursor: pointer; transition: all 0.18s ease; }}
    .seg-btn.active {{ background: rgba(255, 255, 255, 0.94); color: var(--blue); box-shadow: 0 4px 14px rgba(124, 58, 237, 0.16); }}

    .stage-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: stretch; }}
    .stage-grid.only-map, .stage-grid.only-video {{ grid-template-columns: 1fr; }}
    .stage-grid.only-map .video-card {{ display: none; }}
    .stage-grid.only-video .map-card {{ display: none; }}
    @media (max-width: 1080px) {{ .stage-grid {{ grid-template-columns: 1fr; }} }}

    .stage-card {{
      background: rgba(255, 255, 255, 0.56); border: 1px solid rgba(255, 255, 255, 0.88);
      border-radius: 22px; padding: 16px; display: flex; flex-direction: column; gap: 12px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.94);
    }}
    .card-bar {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; }}
    .card-label {{ font-size: 13px; font-weight: 700; color: var(--ink); }}
    .map-box {{ width: 100%; height: 440px; border-radius: 16px; overflow: hidden; border: 1px solid rgba(124, 58, 237, 0.20); position: relative; z-index: 1; background: #F3ECFA; }}
    .map-box.map-light .leaflet-tile-pane {{ filter: saturate(0.45) brightness(1.06) contrast(0.94); }}
    .map-box.map-dark .leaflet-tile-pane {{ filter: invert(100%) hue-rotate(225deg) brightness(90%) contrast(94%) saturate(0.90); }}
    .stage-grid.only-map .map-box {{ height: 540px; }}
    .video-box {{
      width: 100%; height: 440px; border-radius: 16px; overflow: hidden;
      border: 1px solid rgba(249, 115, 22, 0.22); background: #F4ECD8;
      position: relative; display: flex; align-items: center; justify-content: center;
    }}
    .stage-grid.only-video .video-box {{ height: 540px; }}
    #simVideo {{ width: 100%; height: 100%; object-fit: contain; display: block; }}

    .sim-bar {{
      display: flex; flex-wrap: wrap; align-items: center; gap: 14px;
      padding: 14px 20px; border-radius: 18px;
      background: rgba(255, 255, 255, 0.62); border: 1px solid rgba(255, 255, 255, 0.88);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.95);
    }}
    .btn {{
      font-family: inherit; font-size: 12.5px; font-weight: 600;
      border: 1px solid rgba(255, 255, 255, 0.9); background: rgba(255, 255, 255, 0.78);
      color: var(--ink); padding: 8px 15px; border-radius: 999px; cursor: pointer;
      transition: all 0.18s ease; box-shadow: 0 2px 8px rgba(49, 20, 96, 0.06);
    }}
    .btn:hover {{ background: #fff; transform: translateY(-1px); color: var(--blue); }}
    .btn-primary {{ background: linear-gradient(135deg, #7C3AED, #F97316); color: #fff; border-color: transparent; }}
    .btn-primary:hover {{ color: #fff; filter: brightness(1.06); }}
    .chip-btn {{
      font-family: inherit; font-size: 11.5px; font-weight: 600; padding: 5px 11px;
      border-radius: 999px; cursor: pointer; border: 1px solid rgba(124, 58, 237, 0.24);
      background: rgba(245, 240, 255, 0.88); color: var(--blue); transition: all 0.16s ease;
    }}
    .chip-btn:hover, .chip-btn.active {{ background: linear-gradient(135deg, #7C3AED, #F97316); color: #fff; border-color: transparent; }}
    select, input[type="search"] {{
      font-family: inherit; font-size: 12.5px; color: var(--ink);
      background: rgba(255, 255, 255, 0.82); border: 1px solid rgba(255, 255, 255, 0.92);
      border-radius: 999px; padding: 7px 13px; outline: none;
      box-shadow: 0 2px 8px rgba(49, 20, 96, 0.05);
    }}
    #scenarioSel {{ font-weight: 600; color: var(--blue); border-color: rgba(124, 58, 237, 0.28); }}
    .scrub-wrap {{ flex: 1; min-width: 180px; display: flex; align-items: center; gap: 10px; }}
    .scrub-wrap input[type="range"] {{ flex: 1; accent-color: var(--sky); cursor: pointer; }}
    .sync-clock {{ font-size: 21px; font-weight: 800; font-variant-numeric: tabular-nums; min-width: 60px; color: var(--blue); }}
    .quick-jumps {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}

    .inspector {{
      display: none; padding: 20px 24px; border-radius: 20px;
      background: rgba(255, 255, 255, 0.68); border: 1px solid rgba(249, 115, 22, 0.28);
      gap: 20px; align-items: center;
    }}
    .inspector.open {{ display: grid; grid-template-columns: 1fr 1.6fr; }}
    @media (max-width: 820px) {{ .inspector.open {{ grid-template-columns: 1fr; }} }}

    .monitor-panel {{ padding: 26px 28px; display: flex; flex-direction: column; gap: 22px; }}
    .tab-body {{ display: none; }}
    .tab-body.active {{ display: block; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; }}
    @media (max-width: 920px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .subcard {{
      background: rgba(255, 255, 255, 0.56); border: 1px solid rgba(255, 255, 255, 0.88);
      border-radius: 20px; padding: 20px 22px; box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.94);
    }}
    .subcard-title {{ font-size: 14px; font-weight: 700; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .subcard-note {{ font-size: 12px; font-weight: 500; color: var(--ink-soft); }}

    .bar-row {{ margin-bottom: 12px; padding: 6px 8px; border-radius: 10px; cursor: pointer; transition: background 0.15s ease; }}
    .bar-row:hover {{ background: rgba(124, 58, 237, 0.08); }}
    .bar-head {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px; }}
    .bar-track {{ position: relative; height: 8px; border-radius: 999px; background: rgba(49, 20, 96, 0.09); }}
    .bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7C3AED, #C026D3, #F97316); }}
    .bar-obs {{ position: absolute; top: -3px; width: 3px; height: 14px; border-radius: 2px; background: var(--emerald); }}

    .table-tools {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }}
    .table-wrap {{ overflow-x: auto; max-height: 420px; overflow-y: auto; border-radius: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 11px 13px; border-bottom: 1px solid var(--hairline); }}
    th {{
      font-size: 11px; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase;
      color: var(--ink-soft); position: sticky; top: 0; cursor: pointer; user-select: none;
      background: rgba(248, 243, 255, 0.94); backdrop-filter: blur(12px);
    }}
    tbody tr {{ cursor: pointer; transition: background 0.14s ease; }}
    tbody tr:hover {{ background: rgba(249, 115, 22, 0.08); }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }}
    .b-ok {{ background: rgba(5, 150, 105, 0.14); color: var(--emerald); }}
    .b-blue {{ background: rgba(124, 58, 237, 0.14); color: var(--blue); }}
    .b-warn {{ background: rgba(249, 115, 22, 0.18); color: var(--cyan); }}
    .b-err {{ background: rgba(225, 29, 72, 0.14); color: var(--rose); }}
    .b-off {{ background: rgba(49, 20, 96, 0.10); color: var(--ink-soft); }}
    .meta-list {{ margin: 0; padding-left: 18px; display: flex; flex-direction: column; gap: 7px; font-size: 13px; }}
    footer {{ padding: 4px 8px; font-size: 12.5px; color: var(--ink-soft); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; }}
    .leaflet-popup-content-wrapper {{
      background: rgba(255, 255, 255, 0.92); backdrop-filter: blur(16px);
      border-radius: 14px; box-shadow: 0 12px 30px rgba(49, 20, 96, 0.18); color: var(--ink);
    }}
  </style>
</head>
<body>
<div class="wrap">
  <header class="glass">
    <div>
      <div class="brand-row">
        <h1 class="brand-title">STOA</h1>
        <span class="brand-tag" id="hdrSiteTag">Pedestrian Flow Intelligence</span>
      </div>
      <div class="brand-sub" id="hdrSub">Loading model telemetry…</div>
    </div>
    <div class="pills">
      <select id="scenarioSel" aria-label="Select city or scenario"></select>
      <div class="pills" id="hdrPills"></div>
    </div>
  </header>

  <section class="kpi-grid" id="kpiGrid"></section>

  <section class="glass stage-panel">
    <div class="section-head">
      <div class="section-title" id="stageTitle">Real Map &amp; Live Prediction Simulation</div>
      <div class="segmented" role="tablist">
        <button class="seg-btn active" data-layout="split">Split View</button>
        <button class="seg-btn" data-layout="only-map">Interactive Map</button>
        <button class="seg-btn" data-layout="only-video">MP4 Simulation</button>
      </div>
    </div>

    <div class="stage-grid" id="stageGrid">
      <div class="stage-card map-card">
        <div class="card-bar">
          <span class="card-label" id="mapLabel">Real Interactive Map</span>
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
            <select id="basemapSel" aria-label="Basemap style">
              <option value="voyager" selected>Real Street Atlas</option>
              <option value="positron">Clean Light Map</option>
              <option value="dark">Blueprint Dark</option>
            </select>
            <button id="btnLandmarks" class="chip-btn active">Landmarks</button>
            <button id="btnObserved" class="chip-btn active">Observed Rings</button>
            <button id="btnResetMap" class="chip-btn">Fit Bounds</button>
          </div>
        </div>
        <div id="realMap" class="map-box"></div>
      </div>

      <div class="stage-card video-card">
        <div class="card-bar">
          <span class="card-label">STOA Rendered Flow Simulation (MP4)</span>
          <span class="subcard-note" id="livePhaseBadge">17:00 · Evening Peak</span>
        </div>
        <div class="video-box">
          <video id="simVideo" autoplay muted loop playsinline preload="auto"></video>
          <div id="videoMissing" style="display:none;text-align:center;padding:24px;color:var(--ink-soft);">
            <strong>No MP4 simulation rendered for this scenario yet.</strong><br>
            Explore the live interactive map or run <code>src.simulation.marauders_chronicle.render</code>.
          </div>
        </div>
      </div>
    </div>

    <div class="sim-bar">
      <button id="btnPlay" class="btn btn-primary">Pause</button>
      <div class="sync-clock" id="liveClock">17:00</div>
      <div class="scrub-wrap">
        <input id="timeRange" type="range" min="0" max="23.95" step="0.05" value="17" />
        <span style="font-size:12px;color:var(--ink-soft);font-variant-numeric:tabular-nums;">23:55</span>
      </div>
      <select id="selSpeed" aria-label="Playback speed">
        <option value="0.5">0.5×</option>
        <option value="1" selected>1×</option>
        <option value="1.5">1.5×</option>
        <option value="2">2×</option>
      </select>
      <div class="quick-jumps">
        <button class="chip-btn jump-btn" data-hour="3">03h Night</button>
        <button class="chip-btn jump-btn" data-hour="8">08h Commute</button>
        <button class="chip-btn jump-btn" data-hour="13">13h Midday</button>
        <button class="chip-btn jump-btn" data-hour="17">17h Peak</button>
        <button class="chip-btn jump-btn" data-hour="20">20h Dining</button>
      </div>
    </div>

    <div class="inspector" id="sensorInspector">
      <div>
        <div class="section-title" style="margin-bottom:4px;">Selected Sensor Inspector</div>
        <div style="font-size:18px;font-weight:700;margin-bottom:6px;" id="inspName">—</div>
        <div style="font-size:13px;color:var(--ink-soft);margin-bottom:10px;" id="inspMeta">—</div>
        <button id="btnCloseInsp" class="chip-btn">Close Inspector</button>
      </div>
      <div>
        <div class="subcard-title" style="margin-bottom:6px;">
          <span>24-Hour Sensor Profile (Purple: Predicted · Green dashed: Observed)</span>
        </div>
        <svg id="inspSvg" viewBox="0 0 540 120" style="width:100%;height:auto;"></svg>
      </div>
    </div>
  </section>

  <section class="glass monitor-panel">
    <div class="section-head">
      <div class="section-title">Analytics &amp; Model Monitoring</div>
      <div class="segmented" role="tablist">
        <button class="seg-btn active" data-tab="tabAcc">Forecast &amp; Accuracy</button>
        <button class="seg-btn" data-tab="tabLive">Live Hour Leaderboard</button>
        <button class="seg-btn" data-tab="tabHealth">Sensor Health &amp; Drift</button>
        <button class="seg-btn" data-tab="tabBias">Bias Slicing &amp; Pipeline</button>
      </div>
    </div>

    <div id="tabAcc" class="tab-body active">
      <div class="grid-2" style="margin-bottom:22px;">
        <div class="subcard">
          <div class="subcard-title">
            <span>24-Hour City Pulse</span>
            <span class="subcard-note">Purple: predicted · Green dashed: observed</span>
          </div>
          <svg id="pulseSvg" viewBox="0 0 560 210" style="width:100%;height:auto;"></svg>
        </div>
        <div class="subcard">
          <div class="subcard-title">
            <span>Hourly WAPE (%) Across 24 Hours</span>
            <span class="subcard-note">Click any bar to jump to that hour</span>
          </div>
          <svg id="wapeSvg" viewBox="0 0 560 210" style="width:100%;height:auto;"></svg>
        </div>
      </div>
      <div class="subcard">
        <div class="subcard-title">
          <span>Sensor Daily Predicted vs. Observed</span>
          <span class="subcard-note">Click any sensor dot to locate &amp; inspect on the real map</span>
        </div>
        <svg id="scatterSvg" viewBox="0 0 960 220" style="width:100%;height:auto;"></svg>
      </div>
    </div>

    <div id="tabLive" class="tab-body">
      <div class="subcard">
        <div class="subcard-title">
          <span>Busiest Locations at <span id="lbClock">17:00</span></span>
          <span class="subcard-note">Click any row to fly to location on map · Bar: predicted · Green tick: observed</span>
        </div>
        <div id="topSensors"></div>
      </div>
    </div>

    <div id="tabHealth" class="tab-body">
      <div class="subcard">
        <div class="table-tools">
          <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <input id="sensorSearch" type="search" placeholder="Search sensor or location…" />
            <select id="statusFilter">
              <option value="ALL">All statuses</option>
              <option value="Healthy">Healthy</option>
              <option value="High Drift">High Drift</option>
              <option value="Partial Outage">Partial Outage</option>
              <option value="Offline">Offline</option>
              <option value="Forecast Only">Forecast Only</option>
            </select>
          </div>
          <span id="healthSummary" class="subcard-note"></span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th data-sort="name">Location</th>
                <th data-sort="status">Status</th>
                <th data-sort="daily_pred">Daily Pred</th>
                <th data-sort="daily_obs">Daily Obs</th>
                <th data-sort="peak_pred">Peak / hr</th>
                <th data-sort="wape">WAPE</th>
                <th data-sort="bias">Bias</th>
                <th data-sort="missing_hours">Missing Hrs</th>
              </tr>
            </thead>
            <tbody id="sensorTableBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div id="tabBias" class="tab-body">
      <div class="grid-2">
        <div class="subcard">
          <div class="table-tools">
            <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
              <select id="biasDim" aria-label="Slice dimension"></select>
              <select id="biasMit" aria-label="Mitigation">
                <option value="calibrated">calibrated</option>
                <option value="none">none (unmitigated)</option>
              </select>
            </div>
            <span id="biasMeta" class="subcard-note"></span>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Slice Group</th><th>WAPE</th><th>Bias</th><th>MAE</th><th>Flag</th></tr>
              </thead>
              <tbody id="biasTableBody"></tbody>
            </table>
          </div>
        </div>
        <div class="subcard" id="pipeDetails"></div>
      </div>
    </div>
  </section>

  <footer>
    <span>STOA · Planning guidance, not a crowd-safety assessment.</span>
    <span id="footerGen"></span>
  </footer>
</div>

<script>
const BUNDLE = {data_json};
let DATA = BUNDLE.scenarios[0];

const fmt = (n) => n == null || Number.isNaN(n) ? "—" : Math.round(n).toLocaleString();
const fmtPct = (v) => v == null || Number.isNaN(v) ? "—" : (v * 100).toFixed(1) + "%";
function dayPhase(h) {{
  h = ((h % 24) + 24) % 24;
  if (h < 5) return "Late Night";
  if (h < 7) return "Early Dawn";
  if (h < 10) return "Morning Commute";
  if (h < 12) return "Late Morning";
  if (h < 14.5) return "Midday Rush";
  if (h < 16.5) return "Afternoon";
  if (h < 19.5) return "Evening Peak";
  if (h < 22) return "Evening Dining";
  return "Nightfall";
}}
function interp(arr, h) {{
  if (!arr || !arr.length) return null;
  const h0 = Math.floor(h) % 24, h1 = (h0 + 1) % 24, f = h - Math.floor(h);
  const v0 = arr[h0], v1 = arr[h1];
  if (v0 == null || v1 == null) return null;
  return v0 * (1 - f) + v1 * f;
}}

// Populate Scenario / City Selector
const scenarioSel = document.getElementById("scenarioSel");
scenarioSel.innerHTML = BUNDLE.scenarios.map((s, idx) =>
  `<option value="${{idx}}">${{s.label}}</option>`
).join("");
scenarioSel.addEventListener("change", () => loadScenario(parseInt(scenarioSel.value, 10)));

// Layout switcher
const stageGrid = document.getElementById("stageGrid");
document.querySelectorAll("[data-layout]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("[data-layout]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    stageGrid.classList.remove("only-map", "only-video");
    if (btn.dataset.layout !== "split") stageGrid.classList.add(btn.dataset.layout);
    setTimeout(() => {{ if (map) map.invalidateSize(); }}, 120);
  }});
}});

// Monitoring Tabs
document.querySelectorAll("[data-tab]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll("[data-tab]").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  }});
}});

// Universal Real Interactive Map (Leaflet)
let map = null;
let currentTile = null;
let showLandmarks = true;
let showObserved = true;
let predLayer = null;
let obsLayer = null;
let landmarkLayer = null;
let predCircles = [];
let obsCircles = [];
let peakSensorAll = 1;

const OSM_TILE_URL = "https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png";
const OSM_ATTR = "&copy; <a href='https://www.openstreetmap.org/copyright' target='_blank' rel='noopener'>OpenStreetMap</a> contributors";

function applyMapStyle(styleKey) {{
  const el = document.getElementById("realMap");
  el.classList.remove("map-light", "map-dark");
  if (styleKey === "positron") el.classList.add("map-light");
  else if (styleKey === "dark") el.classList.add("map-dark");
}}

function initMapBase() {{
  if (typeof L === "undefined") {{
    document.getElementById("realMap").innerHTML =
      '<div style="padding:24px;color:var(--ink-soft);">Connect to the internet to load real street map tiles.</div>';
    return;
  }}
  map = L.map("realMap", {{ zoomControl: true, preferCanvas: true }}).setView(DATA.map_center, 15);
  currentTile = L.tileLayer(OSM_TILE_URL, {{ attribution: OSM_ATTR, maxZoom: 19 }}).addTo(map);
  obsLayer = L.layerGroup().addTo(map);
  predLayer = L.layerGroup().addTo(map);
  landmarkLayer = L.layerGroup().addTo(map);

  document.getElementById("basemapSel").addEventListener("change", (e) => {{
    applyMapStyle(e.target.value);
  }});
  document.getElementById("btnLandmarks").addEventListener("click", (e) => {{
    showLandmarks = !showLandmarks;
    e.target.classList.toggle("active", showLandmarks);
    if (showLandmarks) landmarkLayer.addTo(map); else map.removeLayer(landmarkLayer);
  }});
  document.getElementById("btnObserved").addEventListener("click", (e) => {{
    showObserved = !showObserved;
    e.target.classList.toggle("active", showObserved);
    obsCircles.forEach(c => c.setStyle({{ opacity: showObserved ? 0.92 : 0 }}));
  }});
  document.getElementById("btnResetMap").addEventListener("click", () => {{
    if (map) map.fitBounds(DATA.map_bounds, {{ padding: [28, 28], maxZoom: 16 }});
  }});
}}

function populateMapLayers() {{
  if (!map) return;
  obsLayer.clearLayers();
  predLayer.clearLayers();
  landmarkLayer.clearLayers();
  predCircles = [];
  obsCircles = [];
  peakSensorAll = Math.max(...DATA.sensors.map(s => s.peak_pred), 1);

  const catCol = {{ transit: "#7C3AED", civic: "#F97316", park: "#059669" }};
  (DATA.landmarks || []).forEach(lm => {{
    const col = catCol[lm.category] || "#7C3AED";
    L.circleMarker([lm.lat, lm.lon], {{
      radius: 6, color: "#fff", weight: 2, fillColor: col, fillOpacity: 0.95
    }}).bindTooltip(`<b>${{lm.name}}</b> (${{lm.category}})`, {{ direction: "top", offset: [0, -6] }}).addTo(landmarkLayer);
  }});

  DATA.sensors.forEach((s, idx) => {{
    const obsC = L.circleMarker([s.lat, s.lon], {{
      radius: 6, color: "#059669", weight: 2, dashArray: "4 3", fillOpacity: 0, interactive: false
    }}).addTo(obsLayer);
    obsCircles.push(obsC);

    const predC = L.circleMarker([s.lat, s.lon], {{
      radius: 7, color: "#6D28D9", weight: 1.8, fillColor: "#8B5CF6", fillOpacity: 0.36
    }}).addTo(predLayer);
    predC.on("click", () => openSensorInspector(idx, false));
    predCircles.push(predC);
  }});

  map.fitBounds(DATA.map_bounds, {{ padding: [28, 28], maxZoom: 16 }});
}}

function updateMapForHour(simHour) {{
  if (!map) return;
  const curPred = DATA.sensors.map(s => interp(s.predicted, simHour) || 0);
  const curObs = DATA.sensors.map(s => interp(s.observed, simHour));
  DATA.sensors.forEach((s, i) => {{
    const p = curPred[i], o = curObs[i];
    const ratio = Math.min(1, p / peakSensorAll);
    const rp = 5 + 18 * Math.sqrt(ratio);
    const fill = ratio > 0.65 ? "#F97316" : ratio > 0.35 ? "#7C3AED" : "#A78BFA";
    predCircles[i].setRadius(rp);
    predCircles[i].setStyle({{ fillColor: fill, color: s.status === "High Drift" ? "#E11D48" : (ratio > 0.65 ? "#EA580C" : "#6D28D9") }});
    predCircles[i].bindPopup(
      `<strong>${{s.name}}</strong><br>`
      + `Predicted: <b style="color:#7C3AED;">${{fmt(p)}} / hr</b><br>`
      + `Observed: <b style="color:#059669;">${{o != null ? fmt(o) + " / hr" : "Forecast-only"}}</b><br>`
      + `Status: <b>${{s.status}}</b>` + (s.wape != null ? ` · Daily WAPE: ${{fmtPct(s.wape)}}` : "")
    );
    if (o != null && showObserved) {{
      obsCircles[i].setRadius(5 + 18 * Math.sqrt(Math.min(1, o / peakSensorAll)));
      obsCircles[i].setStyle({{ opacity: 0.92 }});
    }} else {{
      obsCircles[i].setStyle({{ opacity: 0 }});
    }}
  }});
}}

function openSensorInspector(idx, flyTo = true) {{
  const s = DATA.sensors[idx];
  if (!s) return;
  const box = document.getElementById("sensorInspector");
  box.classList.add("open");
  document.getElementById("inspName").textContent = s.full_name;
  document.getElementById("inspMeta").innerHTML =
    `Status: <b>${{s.status}}</b> · Daily Pred: <b>${{fmt(s.daily_pred)}}</b> · Daily Obs: <b>${{fmt(s.daily_obs)}}</b> · WAPE: <b>${{fmtPct(s.wape)}}</b> · Peak: <b>${{fmt(s.peak_pred)}}/hr @${{String(s.peak_hour).padStart(2,"0")}}h</b>`;

  const maxV = Math.max(...s.predicted, ...s.observed.filter(v => v != null), 1);
  const ptsP = s.predicted.map((v, h) => `${{(26 + 494 * h / 23).toFixed(1)}},${{(96 - 78 * v / maxV).toFixed(1)}}`).join(" ");
  const obsPairs = s.observed.map((v, h) => v != null ? `${{(26 + 494 * h / 23).toFixed(1)}},${{(96 - 78 * v / maxV).toFixed(1)}}` : null).filter(Boolean).join(" ");
  document.getElementById("inspSvg").innerHTML = `
    <line x1="26" y1="96" x2="520" y2="96" stroke="rgba(124,58,237,0.28)" />
    ${{obsPairs ? `<polyline fill="none" stroke="#059669" stroke-width="2" stroke-dasharray="4 3" points="${{obsPairs}}" />` : ""}}
    <polyline fill="none" stroke="#7C3AED" stroke-width="2.6" points="${{ptsP}}" />
    <text x="26" y="112" font-size="10" fill="#5E4E7A">00h</text>
    <text x="150" y="112" font-size="10" fill="#5E4E7A">06h</text>
    <text x="273" y="112" font-size="10" fill="#5E4E7A">12h</text>
    <text x="396" y="112" font-size="10" fill="#5E4E7A">18h</text>
    <text x="506" y="112" font-size="10" fill="#5E4E7A">23h</text>`;

  if (flyTo && map && predCircles[idx]) {{
    map.flyTo([s.lat, s.lon], 17, {{ duration: 0.65 }});
    predCircles[idx].openPopup();
  }}
}}
document.getElementById("btnCloseInsp").addEventListener("click", () => {{
  document.getElementById("sensorInspector").classList.remove("open");
}});

// Synchronized MP4 Video + Real Map + Live KPIs
const simVideo = document.getElementById("simVideo");
const btnPlay = document.getElementById("btnPlay");
const timeRange = document.getElementById("timeRange");
const selSpeed = document.getElementById("selSpeed");
let currentHour = 17.0;
let isScrubbing = false;

function jumpToHour(h) {{
  h = Math.max(0, Math.min(23.95, h));
  timeRange.value = h.toFixed(2);
  if (DATA.has_video && simVideo.duration && Number.isFinite(simVideo.duration)) {{
    simVideo.currentTime = (h / 24.0) * simVideo.duration;
  }}
  updateAllLive(h);
}}

function updateAllLive(simHour) {{
  currentHour = ((simHour % 24) + 24) % 24;
  const hh = Math.floor(currentHour) % 24;
  const mm = Math.floor((currentHour - Math.floor(currentHour)) * 60 / 5) * 5;
  const tStr = `${{String(hh).padStart(2, "0")}}:${{String(mm).padStart(2, "0")}}`;
  document.getElementById("liveClock").textContent = tStr;
  document.getElementById("lbClock").textContent = tStr;
  document.getElementById("livePhaseBadge").textContent = `${{tStr}} · ${{dayPhase(currentHour)}}`;

  const hrRow = DATA.hourly[hh];
  document.getElementById("kpiPeak").textContent = `${{fmt(hrRow.peak_sensor)}} / hr`;
  document.getElementById("kpiMean").textContent = `${{fmt(hrRow.mean_sensor)}} / hr`;
  document.getElementById("kpiAct").textContent = `${{hrRow.activity_pct.toFixed(0)}}%`;

  const cursor = document.getElementById("pulseCursor");
  if (cursor) {{
    const cx = 42 + (498 * (currentHour / 23.0));
    cursor.setAttribute("x1", cx.toFixed(1));
    cursor.setAttribute("x2", cx.toFixed(1));
  }}

  updateMapForHour(currentHour);

  const curPred = DATA.sensors.map(s => interp(s.predicted, currentHour) || 0);
  const curObs = DATA.sensors.map(s => interp(s.observed, currentHour));
  const ranked = curPred.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
  const maxV = Math.max(ranked[0] ? ranked[0][0] : 1, 1);
  document.getElementById("topSensors").innerHTML = ranked.map(([pVal, idx]) => {{
    const s = DATA.sensors[idx];
    const oVal = curObs[idx];
    const pw = Math.min(100, (pVal / maxV) * 100).toFixed(1);
    const ow = oVal != null ? Math.min(100, (oVal / maxV) * 100).toFixed(1) : null;
    return `
      <div class="bar-row" onclick="openSensorInspector(${{idx}}, true)">
        <div class="bar-head">
          <span><strong>${{s.name}}</strong></span>
          <span><strong style="color:var(--blue);">${{fmt(pVal)}}</strong> / <span style="color:var(--emerald);font-weight:600;">${{oVal != null ? fmt(oVal) : "—"}}</span></span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${{pw}}%;"></div>
          ${{ow != null ? `<div class="bar-obs" style="left:calc(${{ow}}% - 1.5px);"></div>` : ""}}
        </div>
      </div>`;
  }}).join("");
}}

simVideo.addEventListener("timeupdate", () => {{
  if (!isScrubbing && simVideo.duration && Number.isFinite(simVideo.duration)) {{
    const h = (simVideo.currentTime / simVideo.duration) * 24.0;
    timeRange.value = Math.min(23.95, h).toFixed(2);
    updateAllLive(h);
  }}
}});
simVideo.addEventListener("play", () => {{ btnPlay.textContent = "Pause"; }});
simVideo.addEventListener("pause", () => {{ btnPlay.textContent = "Play"; }});
btnPlay.addEventListener("click", () => {{
  if (simVideo.paused) simVideo.play().catch(() => {{}});
  else simVideo.pause();
}});
selSpeed.addEventListener("change", () => {{ simVideo.playbackRate = parseFloat(selSpeed.value); }});
timeRange.addEventListener("input", () => {{ isScrubbing = true; jumpToHour(parseFloat(timeRange.value)); }});
timeRange.addEventListener("change", () => {{ isScrubbing = false; }});
document.querySelectorAll(".jump-btn").forEach(btn => {{
  btn.addEventListener("click", () => jumpToHour(parseFloat(btn.dataset.hour)));
}});

function buildPulseChart() {{
  const usePaired = DATA.has_observed;
  const valsP = DATA.hourly.map(r => usePaired ? r.sum_predicted_paired : r.sum_predicted_all);
  const valsO = DATA.hourly.map(r => r.sum_observed_paired);
  const maxY = Math.max(...valsP, ...(usePaired ? valsO : [0]), 1);
  const ptsP = DATA.hourly.map((r, i) => `${{(42 + 498 * r.hour / 23).toFixed(1)}},${{(176 - 148 * valsP[i] / maxY).toFixed(1)}}`).join(" ");
  const ptsO = DATA.hourly.map((r, i) => `${{(42 + 498 * r.hour / 23).toFixed(1)}},${{(176 - 148 * valsO[i] / maxY).toFixed(1)}}`).join(" ");
  document.getElementById("pulseSvg").innerHTML = `
    <defs>
      <linearGradient id="pulseFill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#7C3AED" stop-opacity="0.34" />
        <stop offset="100%" stop-color="#F97316" stop-opacity="0.04" />
      </linearGradient>
    </defs>
    <line x1="42" y1="176" x2="540" y2="176" stroke="rgba(124,58,237,0.28)" />
    <polygon points="42,176 ${{ptsP}} 540,176" fill="url(#pulseFill)" />
    ${{usePaired ? `<polyline fill="none" stroke="#059669" stroke-width="2" stroke-dasharray="5 4" points="${{ptsO}}" />` : ""}}
    <polyline fill="none" stroke="#7C3AED" stroke-width="2.6" points="${{ptsP}}" />
    <line id="pulseCursor" x1="42" y1="22" x2="42" y2="176" stroke="#F97316" stroke-width="1.8" />
    <text x="42" y="196" font-size="11" fill="#5E4E7A">00h</text>
    <text x="166" y="196" font-size="11" fill="#5E4E7A">06h</text>
    <text x="291" y="196" font-size="11" fill="#5E4E7A">12h</text>
    <text x="415" y="196" font-size="11" fill="#5E4E7A">18h</text>
    <text x="524" y="196" font-size="11" fill="#5E4E7A">23h</text>`;
}}

function buildAccuracyCharts() {{
  if (!DATA.has_observed) {{
    document.getElementById("wapeSvg").innerHTML =
      '<text x="280" y="105" text-anchor="middle" font-size="13" fill="#5E4E7A">Forecast-only scenario (no local observed counts)</text>';
    document.getElementById("scatterSvg").innerHTML =
      '<text x="480" y="110" text-anchor="middle" font-size="13" fill="#5E4E7A">Forecast-only scenario — observed comparison unavailable</text>';
    return;
  }}
  const wapes = DATA.hourly.map(r => r.wape || 0);
  const maxW = Math.max(...wapes, 0.5);
  const bars = DATA.hourly.map(r => {{
    const x = 42 + r.hour * 20.8;
    const h = ((r.wape || 0) / maxW) * 148;
    const col = (r.wape || 0) > 0.35 ? "#F97316" : "#7C3AED";
    return `<rect style="cursor:pointer;" onclick="jumpToHour(${{r.hour}})" x="${{x.toFixed(1)}}" y="${{(176 - h).toFixed(1)}}" width="14" height="${{h.toFixed(1)}}" rx="4" fill="${{col}}" opacity="0.88"><title>${{String(r.hour).padStart(2,"0")}}:00 WAPE ${{fmtPct(r.wape)}} (click to jump)</title></rect>`
      + (r.hour % 3 === 0 ? `<text x="${{(x+7).toFixed(1)}}" y="196" text-anchor="middle" font-size="10.5" fill="#5E4E7A">${{String(r.hour).padStart(2,"0")}}h</text>` : "");
  }}).join("");
  document.getElementById("wapeSvg").innerHTML = `
    <line x1="40" y1="176" x2="544" y2="176" stroke="rgba(124,58,237,0.28)" />
    <text x="4" y="34" font-size="10.5" fill="#5E4E7A">${{(maxW*100).toFixed(0)}}%</text>
    <text x="4" y="176" font-size="10.5" fill="#5E4E7A">0%</text>
    ${{bars}}`;

  const paired = DATA.sensors.map((s, idx) => [s, idx]).filter(([s]) => s.daily_obs != null);
  const maxD = Math.max(...paired.map(([s]) => Math.max(s.daily_pred, s.daily_obs)), 1);
  const dots = paired.map(([s, idx]) => {{
    const x = 52 + (s.daily_obs / maxD) * 880;
    const y = 184 - (s.daily_pred / maxD) * 158;
    const col = s.status === "High Drift" ? "#F97316" : "#7C3AED";
    return `<circle style="cursor:pointer;" onclick="openSensorInspector(${{idx}}, true)" cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="5.2" fill="${{col}}" opacity="0.85"><title>${{s.name}}: Pred ${{fmt(s.daily_pred)}} vs Obs ${{fmt(s.daily_obs)}}</title></circle>`;
  }}).join("");
  document.getElementById("scatterSvg").innerHTML = `
    <line x1="52" y1="184" x2="932" y2="184" stroke="rgba(124,58,237,0.28)" />
    <line x1="52" y1="22" x2="52" y2="184" stroke="rgba(124,58,237,0.28)" />
    <line x1="52" y1="184" x2="932" y2="22" stroke="#F97316" stroke-dasharray="5 4" stroke-width="1.5" />
    ${{dots}}
    <text x="492" y="208" text-anchor="middle" font-size="11.5" fill="#5E4E7A">Observed Daily Count →</text>
    <text x="16" y="106" transform="rotate(-90 16 106)" text-anchor="middle" font-size="11.5" fill="#5E4E7A">Predicted →</text>`;
}}

let sortKey = "daily_pred", sortAsc = false;
function renderSensorTable() {{
  const q = document.getElementById("sensorSearch").value.toLowerCase();
  const st = document.getElementById("statusFilter").value;
  const rows = DATA.sensors
    .map((s, idx) => [s, idx])
    .filter(([s]) => (st === "ALL" || s.status === st) && s.full_name.toLowerCase().includes(q))
    .sort(([a], [b]) => {{
      const va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      return sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
    }});
  document.getElementById("healthSummary").textContent = `Showing ${{rows.length}} of ${{DATA.sensors.length}} locations · Click any row to locate on map`;
  const cls = {{ Healthy: "b-ok", "High Drift": "b-err", "Partial Outage": "b-warn", Offline: "b-off", "Forecast Only": "b-blue" }};
  document.getElementById("sensorTableBody").innerHTML = rows.map(([s, idx]) => `
    <tr onclick="openSensorInspector(${{idx}}, true)">
      <td><strong>${{s.name}}</strong></td>
      <td><span class="badge ${{cls[s.status] || "b-blue"}}">${{s.status}}</span></td>
      <td>${{fmt(s.daily_pred)}}</td>
      <td>${{fmt(s.daily_obs)}}</td>
      <td>${{fmt(s.peak_pred)}} <span style="color:var(--ink-soft);">@${{String(s.peak_hour).padStart(2,"0")}}h</span></td>
      <td>${{fmtPct(s.wape)}}</td>
      <td>${{s.bias != null ? (s.bias >= 0 ? "+" : "") + fmtPct(s.bias) : "—"}}</td>
      <td>${{DATA.has_observed ? `${{s.missing_hours}} / 24` : "—"}}</td>
    </tr>`).join("");
}}
document.getElementById("sensorSearch").addEventListener("input", renderSensorTable);
document.getElementById("statusFilter").addEventListener("change", renderSensorTable);
document.querySelectorAll("th[data-sort]").forEach(th => {{
  th.addEventListener("click", () => {{
    if (sortKey === th.dataset.sort) sortAsc = !sortAsc;
    else {{ sortKey = th.dataset.sort; sortAsc = false; }}
    renderSensorTable();
  }});
}});

function drawBiasTable() {{
  const sel = document.getElementById("biasDim");
  const d = sel.value, m = document.getElementById("biasMit").value;
  const rows = (DATA.bias_slices || []).filter(r => r.dimension === d && r.mitigation === m).sort((a, b) => b.wape - a.wape);
  document.getElementById("biasMeta").textContent = rows.length ? `${{rows.length}} slices · disparity ${{rows[0].disparity_ratio.toFixed(3)}}` : "No bias slice data for this scenario";
  document.getElementById("biasTableBody").innerHTML = rows.map(r => `
    <tr>
      <td><strong>${{r.group}}</strong></td>
      <td>${{fmtPct(r.wape)}}</td>
      <td>${{(r.bias >= 0 ? "+" : "") + fmtPct(r.bias)}}</td>
      <td>${{fmt(r.mae)}}</td>
      <td>${{r.worse ? '<span class="badge b-err">Worse</span>' : '<span class="badge b-ok">OK</span>'}}</td>
    </tr>`).join("");
}}
document.getElementById("biasDim").addEventListener("change", drawBiasTable);
document.getElementById("biasMit").addEventListener("change", drawBiasTable);

function renderPipelineTab() {{
  const ph = DATA.pipeline_health || {{}};
  const srcList = Object.entries(ph.rows_by_source || {{}}).map(([k, v]) => `<li><strong>${{k}}</strong>: ${{Number(v).toLocaleString()}} rows</li>`).join("");
  const assump = (DATA.meta.assumptions || []).map(a => `<li>${{a}}</li>`).join("");
  document.getElementById("pipeDetails").innerHTML = `
    <div class="subcard-title"><span>Pipeline &amp; Model Provenance</span><span class="subcard-note">${{DATA.site_label}}</span></div>
    <ul class="meta-list" style="margin-bottom:16px;">
      <li><strong>Model:</strong> ${{DATA.meta.model}}</li>
      <li><strong>MLflow Run:</strong> <code>${{DATA.meta.run_id}}</code></li>
      <li><strong>Source CRS:</strong> <code>${{DATA.meta.crs}}</code> → WGS84</li>
      <li><strong>Historical Table:</strong> ${{ph.rows ? ph.rows.toLocaleString() + " rows" : "Transfer / scenario mode"}}</li>
      <li><strong>Latest Ingest:</strong> ${{ph.last_hour_local || "—"}}</li>
      <li><strong>Outage Zeros Flagged:</strong> ${{ph.rows_suspect_outage_zero != null ? ph.rows_suspect_outage_zero.toLocaleString() : "—"}}</li>
    </ul>
    <div class="section-title" style="margin-bottom:6px;">Data Sources</div>
    <ul class="meta-list" style="margin-bottom:14px;color:var(--ink-soft);">${{srcList || "<li>Scenario prediction outputs</li>"}}</ul>
    <div class="section-title" style="margin-bottom:6px;">Model Assumptions</div>
    <ul class="meta-list" style="color:var(--ink-soft);">${{assump || "<li>Standard STOA feature pipeline</li>"}}</ul>`;
}}

function loadScenario(idx) {{
  DATA = BUNDLE.scenarios[idx] || BUNDLE.scenarios[0];
  document.getElementById("sensorInspector").classList.remove("open");

  document.getElementById("hdrSiteTag").textContent = `${{DATA.site_label}} · Pedestrian Flow Intelligence`;
  document.getElementById("hdrSub").textContent =
    `${{DATA.meta.model}} · Forecast ${{DATA.meta.date}} · ${{DATA.meta.segments.toLocaleString()}} scored segments`;
  document.getElementById("mapLabel").textContent = `Real Interactive Map · ${{DATA.site_label}}`;
  document.getElementById("footerGen").textContent =
    `Built ${{BUNDLE.generated_at}} · ${{BUNDLE.scenarios.length}} scenario(s) loaded · Run ${{DATA.meta.run_id.slice(0, 10)}}`;

  const pills = [
    {{ color: "var(--emerald)", text: `Run ${{DATA.meta.run_id.slice(0, 8)}}` }},
    {{ color: "var(--blue)", text: DATA.pipeline_health.last_hour_local ? `Latest data: ${{DATA.pipeline_health.last_hour_local.slice(0, 16)}}` : `Forecast: ${{DATA.meta.date}}` }},
    {{ color: "var(--indigo)", text: `${{(DATA.meta.segments / 1000).toFixed(1)}}k segments · ${{DATA.meta.sensors_total}} pts` }},
  ];
  document.getElementById("hdrPills").innerHTML = pills.map(p =>
    `<span class="glass-pill pill"><span class="dot" style="background:${{p.color}};"></span>${{p.text}}</span>`
  ).join("");

  const healthVal = DATA.has_observed
    ? `${{DATA.status_counts.Healthy}} / ${{DATA.meta.sensors_in_view}}`
    : `${{DATA.meta.sensors_in_view}} active`;
  const healthSub = DATA.has_observed
    ? `${{DATA.status_counts["High Drift"]}} drift · ${{DATA.status_counts["Partial Outage"] + DATA.status_counts.Offline}} out`
    : "Forecast-only (no ground truth)";

  document.getElementById("kpiGrid").innerHTML = [
    {{ id: "kpiPeak", label: "Peak Location (Live)", val: "—", sub: `Daily peak at ${{String(DATA.meta.peak_hour).padStart(2,"0")}}:00`, color: "var(--blue)" }},
    {{ id: "kpiMean", label: "Avg Location (Live)", val: "—", sub: `Mean across ${{DATA.site_label}}`, color: "var(--sky)" }},
    {{ id: "kpiAct", label: "Activity Index", val: "—", sub: "Share of daily peak flow", color: "var(--indigo)" }},
    {{ id: "kpiWape", label: "Model WAPE", val: DATA.meta.sensor_wape != null ? fmtPct(DATA.meta.sensor_wape) : "Forecast", sub: DATA.meta.sensor_hours_compared ? `${{DATA.meta.sensor_hours_compared.toLocaleString()}} sensor-hrs` : "No ground truth", color: "var(--cyan)" }},
    {{ id: "kpiMae", label: "Log MAE", val: DATA.meta.sensor_log_mae != null ? DATA.meta.sensor_log_mae.toFixed(3) : "—", sub: "Registered STOA model", color: "var(--blue)" }},
    {{ id: "kpiHealth", label: "Sensor Health", val: healthVal, sub: healthSub, color: "var(--emerald)" }},
  ].map(k => `
    <div class="glass kpi">
      <div class="kpi-head"><span>${{k.label}}</span><span class="dot" style="background:${{k.color}};"></span></div>
      <div class="kpi-val" id="${{k.id}}">${{k.val}}</div>
      <div class="kpi-sub">${{k.sub}}</div>
    </div>`).join("");

  if (DATA.has_video && DATA.video_file) {{
    simVideo.style.display = "block";
    document.getElementById("videoMissing").style.display = "none";
    simVideo.src = DATA.video_file;
    simVideo.play().catch(() => {{}});
    btnPlay.disabled = false;
  }} else {{
    simVideo.pause();
    simVideo.removeAttribute("src");
    simVideo.style.display = "none";
    document.getElementById("videoMissing").style.display = "block";
    btnPlay.disabled = true;
  }}

  const dims = [...new Set((DATA.bias_slices || []).map(r => r.dimension))];
  const biasSel = document.getElementById("biasDim");
  biasSel.innerHTML = dims.map(d => `<option value="${{d}}">${{d}}</option>`).join("");
  if (dims.includes("precinct")) biasSel.value = "precinct";

  populateMapLayers();
  buildPulseChart();
  buildAccuracyCharts();
  renderSensorTable();
  drawBiasTable();
  renderPipelineTab();
  updateAllLive(DATA.meta.peak_hour || 17.0);
  timeRange.value = (DATA.meta.peak_hour || 17.0).toFixed(2);
}}

initMapBase();
loadScenario(0);
</script>
</body>
</html>
"""


def _find_scenario_video(sim_dir: Path, fallback_video: Path | None) -> Path | None:
    mp4s = sorted(sim_dir.glob("*.mp4"))
    if mp4s:
        return mp4s[0]
    if sim_dir.name == "marauders_map" and fallback_video and fallback_video.exists():
        return fallback_video
    return None


def build_dashboard(
    sim_dir: Path | None = None,
    sim_root: Path = DEFAULT_SIM_ROOT,
    video_path: Path = DEFAULT_VIDEO,
    counts_meta_path: Path = DEFAULT_COUNTS_META,
    bias_csv_path: Path = DEFAULT_BIAS_CSV,
    out_dir: Path = DEFAULT_OUT,
) -> dict[str, Path]:
    """Discover all scenarios/cities under `sim_root` (or `sim_dir`) and build the STOA dashboard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"

    if sim_dir is not None:
        scenario_dirs = [sim_dir]
    else:
        scenario_dirs = discover_scenario_dirs(sim_root)
        if not scenario_dirs:
            raise SystemExit(f"No scenario directories with meta.json + sensors.json found in {sim_root}")

    default_counts = (
        json.loads(counts_meta_path.read_text(encoding="utf-8"))
        if counts_meta_path.exists()
        else None
    )
    default_bias = load_bias_csv(bias_csv_path)

    scenarios: list[dict[str, Any]] = []
    copied_videos: list[Path] = []
    for idx, sdir in enumerate(scenario_dirs):
        meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
        raw_sensors = json.loads((sdir / "sensors.json").read_text(encoding="utf-8"))
        lm_path = sdir / "landmarks.json"
        raw_landmarks = json.loads(lm_path.read_text(encoding="utf-8")) if lm_path.exists() else None

        s_counts_path = sdir / "counts_hourly_meta.json"
        s_counts = (
            json.loads(s_counts_path.read_text(encoding="utf-8"))
            if s_counts_path.exists()
            else (default_counts if sdir.name == "marauders_map" else None)
        )
        s_bias_path = sdir / "slice_errors.csv"
        s_bias = (
            load_bias_csv(s_bias_path)
            if s_bias_path.exists()
            else (default_bias if sdir.name == "marauders_map" else [])
        )

        vid_src = _find_scenario_video(sdir, video_path)
        video_relpath: str | None = None
        if vid_src is not None and vid_src.exists():
            if idx == 0:
                dest = out_dir / "stoa_simulation.mp4"
                video_relpath = "stoa_simulation.mp4"
            else:
                videos_dir.mkdir(parents=True, exist_ok=True)
                dest = videos_dir / f"{sdir.name}.mp4"
                video_relpath = f"videos/{sdir.name}.mp4"
            if vid_src.resolve() != dest.resolve():
                shutil.copy2(vid_src, dest)
            copied_videos.append(dest)

        scenarios.append(
            compute_scenario_payload(
                scenario_id=sdir.name,
                meta=meta,
                raw_sensors=raw_sensors,
                raw_landmarks=raw_landmarks,
                counts_meta=s_counts,
                bias_rows=s_bias,
                video_relpath=video_relpath,
            )
        )

    bundle = {
        "project": "STOA",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "scenarios": scenarios,
    }

    json_path = out_dir / "dashboard_data.json"
    html_path = out_dir / "index.html"
    json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    html_path.write_text(render_html(bundle), encoding="utf-8")

    outputs: dict[str, Path] = {"html": html_path, "json": json_path}
    if copied_videos:
        outputs["video"] = copied_videos[0]
    print(f"wrote {html_path} and {json_path} ({len(scenarios)} scenario(s))")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the universal STOA purple & orange Liquid Glass monitoring & real-map dashboard."
    )
    parser.add_argument("--sim-dir", type=Path, default=None, help="Optional single scenario directory")
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--counts-meta", type=Path, default=DEFAULT_COUNTS_META)
    parser.add_argument("--bias-csv", type=Path, default=DEFAULT_BIAS_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build_dashboard(
        sim_dir=args.sim_dir,
        sim_root=args.sim_root,
        video_path=args.video,
        counts_meta_path=args.counts_meta,
        bias_csv_path=args.bias_csv,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
