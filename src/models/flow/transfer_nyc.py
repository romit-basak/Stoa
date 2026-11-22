"""Cross-city transfer test: the portable flow model, trained on Melbourne, scored on NYC.

Usage: python -m src.models.flow.transfer_nyc --model-uri URI [--weather-dir DIR]
       [--output-dir DIR] [--since YEAR] [--uni-teaching]

NYC DOT's bi-annual counts are evaluation only (CLAUDE.md, "Generalization"): nothing here
fits a model on them. They're period totals, not hourly counts: a weekday 07-09 and 16-19 and
a Saturday 12-14, on both sides of the street, at 114 locations, twice a year. So:

1. each count point is snapped to the NYC walking graph. On a street, the sidewalk on each side
   is found and their predictions are summed, since Melbourne's sensors mostly count one
   sidewalk and NYC counts both. Elsewhere (bridges, the greenway, pedestrian streets) the
   nearest path is used once;
2. the model predicts every hour of each count window, on every day that could have been the
   count day (Tuesday to Thursday for the weekday, any Saturday), with that hour's weather. A
   window's prediction is the sum over its hours, averaged over the candidate days;
3. the Space Syntax and betweenness baselines, fitted on Melbourne's fold sensors exactly as in
   training, are scored the same way;
4. scores: Spearman rank correlation across locations (the team's goal is 0.6, uncalibrated),
   MAE on log totals, the median log ratio (the city-wide offset), and the log MAE after one
   city-wide scale fitted on k locations' counts, over random draws of the k locations.

Calendar features Melbourne has and NYC doesn't are set as labelled assumptions
(configs/transfer.yaml). Writes <output_dir>/transfer/nyc/: units.csv (the snapped segments),
totals.csv (observed and predicted totals per location, round and window), calibration.csv
and summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.ops
import yaml
from scipy.stats import spearmanr

from src.features import counts_hourly as ch
from src.features import shade as sh
from src.features import training_frame as tf
from src.features.common import load_features_config, log
from src.ingest.common import REPO_ROOT, load_config, setup_logging
from src.models.flow import baselines
from src.models.flow import data as flow_data

MONTHS = {"may": 5, "june": 6, "sep": 9, "sept": 9, "oct": 10}
COLUMN = re.compile(r"(may|june|sept|sep|oct)_?(\d{2})_(am|pm|md|p_m)")
ROADS = {
    "residential",
    "living_street",
    "unclassified",
    "tertiary",
    "secondary",
    "primary",
    "trunk",
    "tertiary_link",
    "secondary_link",
    "primary_link",
    "service",
    "pedestrian",
}
PATHS = {"footway", "path", "cycleway", "pedestrian", "steps"}
KEYS = ["loc", "year", "month", "period"]


def load_transfer_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or REPO_ROOT / "configs" / "transfer.yaml") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------------------------


def parse_counts(points: pd.DataFrame) -> pd.DataFrame:
    """One row per (loc, year, month, period) with a count. NYC DOT's column names vary
    (`may_22_p_m`, `oct24_am`, `june_24_pm`), so they're parsed by pattern; 0 means not
    collected and is dropped."""
    rows = []
    for col in points.columns:
        m = COLUMN.fullmatch(col)
        if not m:
            continue
        month, yy, period = m.groups()
        part = pd.DataFrame(
            {
                "loc": points["loc"].astype(int).to_numpy(),
                "year": 2000 + int(yy),
                "month": MONTHS[month],
                "period": "pm" if period == "p_m" else period,
                "observed": pd.to_numeric(points[col], errors="coerce").to_numpy(),
            }
        )
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out = out[out["observed"].gt(0)]
    return out.sort_values(KEYS).reset_index(drop=True)


def count_days(
    year: int, month: int, period: str, weekdays: list[int], holidays: set
) -> list[pd.Timestamp]:
    """The days that could have been the count day: Saturdays for the Saturday window,
    otherwise the listed weekdays that aren't public holidays."""
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[days.month == month]
    if period == "md":
        return list(days[days.dayofweek == 5])
    return [d for d in days if d.dayofweek in weekdays and d.date() not in holidays]


# --------------------------------------------------------------------------------------------
# Snapping
# --------------------------------------------------------------------------------------------


def normalise_name(name: Any) -> str:
    """A street name in a comparable form: "East 42nd Street" and "East 42 St" agree."""
    s = name.lower() if isinstance(name, str) else ""
    for a, b in [
        ("avenue", "ave"),
        ("street", "st"),
        ("boulevard", "blvd"),
        ("road", "rd"),
        ("place", "pl"),
        ("parkway", "pkwy"),
    ]:
        s = s.replace(a, b)
    s = re.sub(r"\b(east|west|north|south)\b", lambda m: m.group(1)[0], s)
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s)
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def _names_agree(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def _axis_deg(line: Any) -> float:
    c = np.asarray(line.coords)
    v = c[-1] - c[0]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _side(street: Any, point: Any, other: Any) -> float:
    """Which side of `street` (at the point nearest `point`) `other` lies on: +1 or -1."""
    at = street.project(point)
    a = street.interpolate(at)
    b = street.interpolate(min(at + 1.0, street.length))
    if b.equals(a):
        a, b = street.interpolate(max(at - 1.0, 0.0)), a
    q = shapely.ops.nearest_points(other, point)[0]
    cross = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x)
    return float(np.sign(cross))


def snap(
    points: gpd.GeoDataFrame, segments: gpd.GeoDataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """The segments each count point is predicted on, both in a metric CRS.

    `points` needs loc, street_nam and geometry; `segments` segment_id, name, highway,
    is_sidewalk, is_crossing and geometry. On a street (the nearest road whose name matches
    the point's street, else the nearest road), the nearest sidewalk on each side that runs
    along it: kind `both_sidewalks`, weight 1 each. Without a sidewalk on both sides, the
    nearest path (or that road) alone: kind `single_path`, weight 1.
    """
    radius = cfg["search_radius_m"]
    sw_radius, sw_angle = cfg["sidewalk_radius_m"], cfg["sidewalk_angle_deg"]
    index = segments.sindex
    rows = []
    for _, p in points.iterrows():
        near = segments.iloc[
            index.query(p.geometry, predicate="dwithin", distance=radius)
        ].copy()
        near["distance_m"] = near.distance(p.geometry)
        near = near[~near["is_crossing"].astype(bool)]
        roads = near[near["highway"].isin(ROADS) & ~near["is_sidewalk"].astype(bool)]
        want = normalise_name(p["street_nam"])
        named = roads[[_names_agree(normalise_name(n), want) for n in roads["name"]]]
        street = (named if len(named) else roads).sort_values("distance_m").head(1)
        picked: list[tuple[Any, str]] = []
        if len(street):
            line = street.iloc[0].geometry
            axis = _axis_deg(line)
            sw = near[
                near["is_sidewalk"].astype(bool) & near["distance_m"].le(sw_radius)
            ].copy()
            if len(sw):
                diff = np.array(
                    [(_axis_deg(g) - axis + 90) % 180 - 90 for g in sw.geometry]
                )
                sw = sw[np.abs(diff) <= sw_angle]
            if len(sw):
                sw["side"] = [_side(line, p.geometry, g) for g in sw.geometry]
                sw = sw[sw["side"] != 0]
                best = sw.sort_values("distance_m").groupby("side").head(1)
                if best["side"].nunique() == 2:
                    picked = [(r, "both_sidewalks") for _, r in best.iterrows()]
        if not picked:
            paths = near[near["highway"].isin(PATHS)].sort_values("distance_m")
            pool = paths if len(paths) else street
            if len(pool):
                picked = [(pool.iloc[0], "single_path")]
        for r, kind in picked:
            rows.append(
                {
                    "loc": int(p["loc"]),
                    "street": p["street_nam"],
                    "segment_id": r["segment_id"],
                    "kind": kind,
                    "weight": 1.0,
                    "distance_m": round(float(r["distance_m"]), 1),
                    "segment_highway": r["highway"],
                    "street_name_matched": bool(len(named)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Hourly rows and features
# --------------------------------------------------------------------------------------------


def hourly_rows(
    counts: pd.DataFrame,
    cfg: dict[str, Any],
    holidays: set,
    tz: str,
) -> pd.DataFrame:
    """Every (loc, year, month, period, day, hour) to predict, with local and UTC stamps."""
    rounds = counts[KEYS].drop_duplicates()
    parts = []
    for (year, month, period), locs in rounds.groupby(["year", "month", "period"])[
        "loc"
    ]:
        days = count_days(year, month, period, cfg["weekdays"], holidays)
        hours = cfg["period_hours"][period]
        stamp = pd.DataFrame(
            [(d, h) for d in days for h in hours], columns=["day", "hour"]
        )
        grid = pd.DataFrame({"loc": locs.unique()}).merge(stamp, how="cross")
        grid["year"], grid["month"], grid["period"] = year, month, period
        parts.append(grid)
    rows = pd.concat(parts, ignore_index=True)
    local = (rows["day"] + pd.to_timedelta(rows["hour"], unit="h")).dt.tz_localize(
        tz, nonexistent="shift_forward", ambiguous=False
    )
    rows["ts_local"] = local
    rows["ts_utc"] = local.dt.tz_convert("UTC")
    return rows


def row_features(
    rows: pd.DataFrame,
    holidays: pd.Series,
    weather: pd.DataFrame,
    center: tuple,
    cfg: dict[str, Any],
    fcfg: dict[str, Any],
    uni_teaching: bool = False,
) -> pd.DataFrame:
    """The hourly table's features for each row, as Melbourne's are built
    (src/features/counts_hourly.py), with the calendar assumptions of configs/transfer.yaml."""
    years = set(range(rows["year"].min() - 1, rows["year"].max() + 2))
    school = (
        ([("1900-01-01", "2100-12-31")], years) if cfg["school_term"] else ([], years)
    )
    if uni_teaching:
        uni = (
            {"teaching": [("1900-01-01", "2100-12-31")], "break": [], "exams": []},
            years,
        )
    else:
        uni = ({"teaching": [], "break": [], "exams": []}, set())
    out = ch.time_features(rows, holidays, school, uni, center, fcfg["counts_hourly"])
    wx = ch.weather_for_counts(
        rows["ts_utc"], weather, fcfg["counts_hourly"]["weather_interval_vars"]
    )
    return pd.concat([out, wx], axis=1)


def design(
    rows: pd.DataFrame,
    units: pd.DataFrame,
    times: pd.DataFrame,
    seg: pd.DataFrame,
    shade_table: pd.DataFrame | None,
    shade_meta: dict[str, Any] | None,
    columns: list[str],
    events: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (hour row, unit segment) with every model column."""
    base = pd.concat(
        [rows, times.drop(columns=["date_local"], errors="ignore")], axis=1
    )
    base["row"] = np.arange(len(base))
    for c in events.columns:
        base[c] = events[c].to_numpy()
    x = base.merge(units[["loc", "segment_id", "weight"]], on="loc", how="inner")
    x = x.join(seg, on="segment_id")
    if shade_table is not None and shade_meta is not None:
        b = sh.bin_of(
            x["sun_elevation_deg"].to_numpy(),
            x["sun_azimuth_deg"].to_numpy(),
            shade_meta,
        )
        look = shade_table.set_index(["segment_id", "sun_bin"])["shade_portable"]
        v = look.reindex(pd.MultiIndex.from_arrays([x["segment_id"], b])).to_numpy()
        has = bool(shade_table["shade_portable"].notna().any())
        v = np.where(np.isnan(v.astype(float)), 0.0 if has else np.nan, v)
        x["shade_portable"] = np.where(b < 0, 1.0, v).astype("float32")
    else:
        x["shade_portable"] = np.float32(np.nan)
    missing = [c for c in columns if c not in x.columns]
    if missing:
        raise ValueError(f"NYC rows lack model columns: {missing[:5]}")
    return x


# --------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------


def window_totals(x: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    """Per (loc, year, month, period): the sum over hours and unit segments of each day,
    averaged over the candidate days."""
    w = x[pred_cols].mul(x["weight"], axis=0)
    w[KEYS + ["day"]] = x[KEYS + ["day"]]
    per_day = w.groupby(KEYS + ["day"])[pred_cols].sum()
    return per_day.groupby(KEYS).mean().reset_index()


def log_error(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    return np.log1p(pred) - np.log1p(obs)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def scores(t: pd.DataFrame, pred: str) -> dict[str, Any]:
    """Uncalibrated scores of one prediction column against `observed`."""
    err = log_error(t[pred].to_numpy(), t["observed"].to_numpy())
    out: dict[str, Any] = {
        "n_obs": len(t),
        "n_locations": int(t["loc"].nunique()),
        "log_mae": float(np.mean(np.abs(err))),
        "median_log_ratio": float(np.median(err)),
        "log_mae_offset_removed": float(np.mean(np.abs(err - np.median(err)))),
    }
    # Rank correlation across locations, per window: each location's mean log count over the
    # rounds scored, so the period (AM against PM) can't inflate it.
    per_period = {}
    for period, g in t.groupby("period"):
        m = g.groupby("loc")[[pred, "observed"]].apply(lambda d: np.log1p(d).mean())
        per_period[period] = spearman(m[pred].to_numpy(), m["observed"].to_numpy())
    out["spearman_by_period"] = per_period
    out["spearman_mean"] = float(np.nanmean(list(per_period.values())))
    by_round = [
        spearman(g[pred].to_numpy(), g["observed"].to_numpy())
        for _, g in t.groupby(["year", "month", "period"])
    ]
    out["spearman_per_round_median"] = float(np.nanmedian(by_round))
    return out


def calibrated(
    t: pd.DataFrame, pred: str, k: int, draws: int, rng: np.random.Generator
) -> dict[str, float]:
    """log MAE on the other locations after one city-wide scale, the mean log ratio over k
    random locations' counts; median and quartiles over the draws."""
    locs = t["loc"].unique()
    err = log_error(t[pred].to_numpy(), t["observed"].to_numpy())
    loc_of = t["loc"].to_numpy()
    out = []
    for _ in range(draws):
        pick = rng.choice(locs, size=k, replace=False)
        fit = np.isin(loc_of, pick)
        scale = err[fit].mean()
        out.append(np.mean(np.abs(err[~fit] - scale)))
    q = np.quantile(out, [0.25, 0.5, 0.75])
    return {"k": k, "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}


def interval_coverage(t: pd.DataFrame) -> float:
    """Share of observed totals inside the summed hourly 80% bounds (a loose check: a sum of
    hourly quantiles is wider than the quantile of the sum)."""
    if t["lower"].isna().all():
        return float("nan")
    inside = t["observed"].between(t["lower"], t["upper"])
    return float(inside.mean())


# --------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------


def fit_baselines(fcfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Both network baselines fitted on every Melbourne fold sensor (the holdout precincts
    stay out, as for the final model), and the portable segment features they read."""
    site = fcfg["counts_hourly"]["site"]
    feats = REPO_ROOT / fcfg["processed_dir"] / "features" / site
    cols = ["location_id", "hour_of_week", "pedestriancount"]
    cols += ["has_location", "location_type", "suspect_outage_zero"]
    hourly = pd.read_parquet(feats / "counts_hourly.parquet", columns=cols)
    hourly = hourly.iloc[tf.eligible(hourly)].reset_index(drop=True)
    splits = pd.read_parquet(
        REPO_ROOT / fcfg["processed_dir"] / "training" / site / "splits.parquet"
    ).set_index("location_id")
    fold = splits["fold"].reindex(hourly["location_id"]).to_numpy()
    rows = np.flatnonzero(np.nan_to_num(fold, nan=-1) >= 0)
    osm = tf.sensor_features(
        pd.read_parquet(feats / "sensor_osm_features.parquet"), "portable"
    )
    y = hourly["pedestriancount"].to_numpy(dtype=float)
    fitted = {
        rung: baselines.NetworkBaseline(baselines.network_columns(rung, osm)).fit(
            hourly, osm, rows, y
        )
        for rung in baselines.RUNGS
    }
    log.info(
        "baselines fitted on %d Melbourne fold sensors",
        len(np.unique(hourly["location_id"].to_numpy()[rows])),
    )
    return fitted, osm


def run(
    model: Any,
    weather_dir: Path,
    since: int | None = None,
    uni_teaching: bool = False,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_transfer_config()
    data_cfg, fcfg = load_config(), load_features_config()
    name = cfg["site"]
    site = data_cfg["sites"][name]
    since = cfg["since_year"] if since is None else since
    raw = REPO_ROOT / data_cfg["raw_dir"]
    feats = REPO_ROOT / fcfg["processed_dir"] / "features" / name

    points = gpd.read_file(raw / cfg["counts"])
    counts = parse_counts(points)
    counts_all = counts
    counts = counts[counts["year"] >= since].reset_index(drop=True)

    seg = gpd.read_parquet(feats / "osm_segment_features.parquet")
    seg = seg[seg["in_feature_mask"]].reset_index(drop=True)
    crs = f"EPSG:{site['utm_epsg']}"
    units = snap(points.to_crs(crs), seg.to_crs(crs), cfg["snap"])
    log.info(
        "snapped %d locations: %s",
        units["loc"].nunique(),
        units.groupby("kind")["loc"].nunique().to_dict(),
    )

    hol = pd.read_csv(raw / "calendars" / f"{name}_public_holidays.csv")["date"]
    holidays = pd.to_datetime(hol)
    rows = hourly_rows(counts, cfg, set(holidays.dt.date), site["timezone"])
    weather = ch.read_weather(weather_dir, name)
    times = row_features(
        rows, holidays, weather, tuple(site["center"]), cfg, fcfg, uni_teaching
    )
    rows_t = pd.concat([rows, times[["hour_of_week"]]], axis=1)
    columns = list(model.metadata.get_input_schema().input_names())
    seg_cols = [c for c in columns if c in seg.columns]
    seg_tab = seg.set_index("segment_id")[seg_cols].astype("float32")
    shade_path = feats / "osm_segment_shade.parquet"
    shade_table = pd.read_parquet(shade_path) if shade_path.exists() else None
    shade_meta = (
        json.loads((feats / "shade_meta.json").read_text())
        if shade_table is not None
        else None
    )
    events = tf.event_columns(
        rows["ts_local"],
        pd.Series(index=rows.index, dtype=object),
        name,
        data_cfg,
        fcfg,
    )
    x = design(
        rows_t.drop(columns=["hour_of_week"]),
        units,
        times,
        seg_tab,
        shade_table,
        shade_meta,
        columns,
        events,
    )
    pred = model.predict(x[columns].astype("float32"))
    if isinstance(pred, pd.DataFrame):
        for c in ("predicted", "lower", "upper"):
            x[c] = pred[c].to_numpy()
    else:
        # A model registered before the interval bundle (v1) predicts the mean only.
        x["predicted"] = np.asarray(pred, dtype="float32").ravel()
        x["lower"] = x["upper"] = np.nan

    fitted, _ = fit_baselines(fcfg)
    unit_osm = seg_tab.copy()
    for rung, b in fitted.items():
        ex = pd.DataFrame(
            {
                "location_id": x["segment_id"].to_numpy(),
                "hour_of_week": x["hour_of_week"].to_numpy(),
            }
        )
        x[f"baseline_{rung}"] = b.predict(ex, unit_osm, np.arange(len(ex)))

    pred_cols = ["predicted", "lower", "upper"] + [f"baseline_{r}" for r in fitted]
    totals = window_totals(x, pred_cols).merge(counts, on=KEYS)
    kinds = units.groupby("loc")["kind"].first()
    totals["kind"] = totals["loc"].map(kinds)

    rng_seed = cfg["calibration"]["seed"]
    summary: dict[str, Any] = {
        "site": name,
        "since_year": since,
        "rounds": sorted(
            f"{y}-{m:02d}"
            for y, m in counts[["year", "month"]]
            .drop_duplicates()
            .itertuples(index=False)
        ),
        "rounds_available": int(
            counts_all[["year", "month"]].drop_duplicates().shape[0]
        ),
        "locations_snapped": units.groupby("kind")["loc"].nunique().to_dict(),
        "uni_teaching": "teaching" if uni_teaching else "unknown",
        "school_term": bool(cfg["school_term"]),
        "subsets": {},
    }
    cal_rows = []
    for subset, t in {
        "both_sidewalks": totals[totals["kind"].eq("both_sidewalks")],
        "all": totals,
    }.items():
        res: dict[str, Any] = {}
        for col, name in [
            ("predicted", "model"),
            ("baseline_space_syntax", "space_syntax"),
            ("baseline_betweenness", "betweenness"),
        ]:
            s = scores(t, col)
            rng = np.random.default_rng(rng_seed)
            curve = [
                calibrated(t, col, k, cfg["calibration"]["draws"], rng)
                for k in cfg["calibration"]["k"]
                if k < t["loc"].nunique()
            ]
            for c in curve:
                cal_rows.append({"subset": subset, "predictor": name, **c})
            head = next(c for c in curve if c["k"] == cfg["calibration"]["headline_k"])
            s["log_mae_calibrated_k"] = cfg["calibration"]["headline_k"]
            s["log_mae_calibrated"] = head["median"]
            res[name] = s
        res["model"]["interval_coverage_summed"] = interval_coverage(t)
        summary["subsets"][subset] = res
    return {
        "summary": summary,
        "units": units,
        "totals": totals,
        "calibration": pd.DataFrame(cal_rows),
    }


def load_model(uri: str) -> Any:
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(uri)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-uri",
        default="models:/stoa-flow-portable@champion",
        help="a portable model bundle: a registry URI or a downloaded model directory",
    )
    ap.add_argument(
        "--weather-dir",
        type=Path,
        default=None,
        help="folder with nyc_openmeteo_<year>.json (default: the raw weather folder)",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--since", type=int, default=None, help="first count year scored")
    ap.add_argument(
        "--uni-teaching",
        action="store_true",
        help="set the university-teaching flag on (sensitivity check)",
    )
    args = ap.parse_args()

    data_cfg = load_config()
    weather_dir = args.weather_dir or REPO_ROOT / data_cfg["raw_dir"] / "weather"
    model = load_model(args.model_uri)
    out = run(model, weather_dir, args.since, args.uni_teaching)
    fcfg = flow_data.flow_config()
    dest = (
        flow_data.output_dir(fcfg, args.output_dir)
        / "transfer"
        / out["summary"]["site"]
    )
    if args.uni_teaching:
        dest = dest / "uni_teaching"
    dest.mkdir(parents=True, exist_ok=True)
    out["summary"]["model_uri"] = args.model_uri
    (dest / "summary.json").write_text(json.dumps(out["summary"], indent=2) + "\n")
    out["units"].to_csv(dest / "units.csv", index=False)
    out["totals"].to_csv(dest / "totals.csv", index=False)
    out["calibration"].to_csv(dest / "calibration.csv", index=False)
    log.info("wrote %s", dest)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
