"""Build the universal STOA Liquid Glass monitoring, real-map, and simulation dashboard.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_ROOT = REPO_ROOT / "data" / "simulation"
DEFAULT_VIDEO = (
    REPO_ROOT
    / "data"
    / "simulation"
    / "marauders_chronicle"
    / "marauders_chronicle.mp4"
)


def _default_counts_meta() -> Path:
    base = REPO_ROOT / "data" / "processed" / "features"
    matches = sorted(base.glob("*/counts_hourly_meta.json"))
    return matches[0] if matches else (base / "counts_hourly_meta.json")


def _default_bias_csv() -> Path:
    base = REPO_ROOT / "data" / "processed" / "bias"
    matches = sorted(base.glob("*/slice_errors.csv"))
    return matches[0] if matches else (base / "slice_errors.csv")


DEFAULT_COUNTS_META = _default_counts_meta()
DEFAULT_BIAS_CSV = _default_bias_csv()
DEFAULT_OUT = REPO_ROOT / "data" / "dashboard"

# Fallback Melbourne CBD landmarks when a Melbourne scenario lacks landmarks.json
MELBOURNE_LANDMARKS_UTM: list[dict[str, Any]] = [
    {
        "name": "Flinders Street Station",
        "category": "transit",
        "x": 321020.0,
        "y": 5812445.0,
    },
    {
        "name": "Southern Cross Station",
        "category": "transit",
        "x": 319820.0,
        "y": 5812455.0,
    },
    {"name": "Melbourne Central", "category": "transit", "x": 320815.0, "y": 5813215.0},
    {
        "name": "Parliament Station",
        "category": "transit",
        "x": 321700.0,
        "y": 5813160.0,
    },
    {"name": "Bourke Street Mall", "category": "civic", "x": 320885.0, "y": 5812908.0},
    {
        "name": "Queen Victoria Market",
        "category": "civic",
        "x": 320320.0,
        "y": 5813665.0,
    },
    {"name": "Town Hall", "category": "civic", "x": 320985.0, "y": 5812780.0},
    {"name": "Flagstaff Gardens", "category": "park", "x": 320045.0, "y": 5813325.0},
    {"name": "Treasury Gardens", "category": "park", "x": 321895.0, "y": 5812655.0},
    {"name": "Birrarung Marr", "category": "park", "x": 321520.0, "y": 5812370.0},
    {"name": "Southbank", "category": "park", "x": 320890.0, "y": 5812145.0},
]


def utm_to_latlon(
    easting: float, northing: float, zone: int, northern: bool
) -> tuple[float, float]:
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
        + (
            61.0
            + 90.0 * t1
            + 298.0 * c1
            + 45.0 * t1**2
            - 252.0 * e_prime2
            - 3.0 * c1**2
        )
        * d**6
        / 720.0
    )
    lon = (
        lon0
        + (
            d
            - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * e_prime2 + 24.0 * t1**2)
            * d**5
            / 120.0
        )
        / cos_p
    )

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
            from pyproj.exceptions import CRSError
        except ImportError:
            tr = None
        else:
            code = f"EPSG:{epsg}" if epsg is not None else str(crs)
            try:
                tr = Transformer.from_crs(code, "EPSG:4326", always_xy=True)
            except (CRSError, ValueError):
                tr = None
        if tr is not None:
            return lambda x, y: (
                round(float(tr.transform(float(x), float(y))[1]), 6),
                round(float(tr.transform(float(x), float(y))[0]), 6),
            )

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
    cleaned = re.sub(
        r"^(I-Hub|iHub|COM Pole \d+\s*-\s*)\s*", "", name, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*-\s*New footpath$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(COM Pole \d+\)$", "", cleaned, flags=re.IGNORECASE)
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
        peak_pred = max(
            (v for v in s.get("predicted", []) if v is not None), default=0.0
        )
        return (1 if obs_count > 0 else 0, peak_pred)

    ordered = sorted(sensors, key=priority, reverse=True)
    kept: list[dict[str, Any]] = []
    for s in ordered:
        if any(
            math.hypot(s["lat"] - k["lat"], s["lon"] - k["lon"]) < min_dist_deg
            for k in kept
        ):
            continue
        kept.append(s)
    return kept


def classify_sensor_status(
    missing_hours: int, wape: float | None, has_any_obs: bool
) -> str:
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


def _infer_site_label(
    scenario_id: str, meta: dict[str, Any], center: tuple[float, float]
) -> str:
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
                "name": short_sensor_name(
                    str(s.get("name", f"Point {len(sensor_records) + 1}"))
                ),
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
    bounds = [
        [round(min_lat, 6), round(min_lon, 6)],
        [round(max_lat, 6), round(max_lon, 6)],
    ]
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
                {
                    "name": lm["name"],
                    "category": lm["category"],
                    "lat": llat,
                    "lon": llon,
                }
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
            "sensor_wape": round(float(sensor_wape), 4)
            if sensor_wape is not None
            else None,
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
            "first_hour_local": counts_meta.get("first_hour_local")
            if counts_meta
            else None,
            "last_hour_local": counts_meta.get("last_hour_local")
            if counts_meta
            else None,
            "rows_suspect_outage_zero": int(
                counts_meta.get("rows_suspect_outage_zero", 0)
            )
            if counts_meta
            else None,
            "rows_without_weather": int(counts_meta.get("rows_without_weather", 0))
            if counts_meta
            else None,
            "rows_by_source": counts_meta.get("rows_by_source", {})
            if counts_meta
            else {},
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
        if (
            child.is_dir()
            and (child / "meta.json").exists()
            and (child / "sensors.json").exists()
        ):
            found.append(child)
    return found


def render_html(bundle: dict[str, Any]) -> str:
    """Render the multi-scenario, multi-city purple & orange Liquid Glass STOA dashboard."""
    data_json = json.dumps(bundle, separators=(",", ":"))
    template_path = Path(__file__).with_name("template.html")
    return template_path.read_text(encoding="utf-8").replace(
        "__STOA_DATA_JSON__", data_json
    )


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
    """Discover scenarios/cities under `sim_root` (or `sim_dir`) and build the STOA dashboard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"

    if sim_dir is not None:
        scenario_dirs = [sim_dir]
    else:
        scenario_dirs = discover_scenario_dirs(sim_root)
        if not scenario_dirs:
            raise SystemExit(
                f"No scenario directories with meta.json + sensors.json found in {sim_root}"
            )

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
        raw_landmarks = (
            json.loads(lm_path.read_text(encoding="utf-8"))
            if lm_path.exists()
            else None
        )

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
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
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
        description=(
            "Build the universal STOA purple & orange Liquid Glass "
            "monitoring & real-map dashboard."
        )
    )
    parser.add_argument(
        "--sim-dir", type=Path, default=None, help="Optional single scenario directory"
    )
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
