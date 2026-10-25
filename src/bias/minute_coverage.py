"""Bias in the per-minute data: which kinds of hours the archive and its profiles cover.

Usage: python -m src.bias.minute_coverage

The within-hour profiles (src/features/minute_profiles.py) keep only sensor-hours whose minutes
add up to the hourly count and that have at least `min_hour_count` people. That filter is
needed for clean shares, but it drops quiet hours on purpose, so the profiles lean towards busy
places and busy times. This step measures by how much, using the same slices as the hourly
table's bias report (configs/bias.yaml): time of day, day type, holidays, weather, era, AFL days,
and the sensor slices (precinct, volume tier, amenity tier, slope band and the rest).

For every group it reports the sensor-hours of the hourly table inside the archive's span, the
share the archive has, the share whose minutes add up, and the share kept for the profiles. A
group is flagged `under` when its kept share is below `low_ratio` times the overall kept share
and it has at least `min_hours` sensor-hours, so a single quiet evening doesn't raise a flag.

Mitigation: pooled profiles. Every hour whose minutes add up, however quiet, is summed with the
others of its volume tier (quiet, medium, busy), day type and local hour, and the shares come
from the sums. A pooled cell is usable once it holds `pooled_min_people` people, so quiet
streets get a profile from many quiet hours together. `pooled_share` is the share of each
group's sensor-hours that land in a usable pooled cell, next to `kept_share` for comparison.

Reads minute_hour_status.parquet (minute_profiles), the archive, the hourly table for the
archive's span, and sensor_slices.csv (bias_slices). Writes data/processed/bias/melbourne/
minute_coverage.csv and minute_coverage_meta.json, and the pooled profiles to
data/processed/features/melbourne/minute_profiles_by_tier.parquet.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.bias.representation import ROW_DIMS, SENSOR_DIMS
from src.bias.slices import afl_days, load_bias_config, row_slices
from src.features import minute_profiles as mp
from src.features.common import (
    config_hash,
    load_features_config,
    log,
    write_meta,
    write_parquet,
)
from src.ingest.common import REPO_ROOT, load_config, setup_logging

SLICE_INPUTS = [
    "location_id",
    "ts_local",
    "date_local",
    "hour_local",
    "is_weekend",
    "is_public_holiday",
    "school_term",
    "uni_teaching",
    "wx_precipitation",
    "wx_temperature_2m",
    "pedestriancount",
]
COLUMNS = [
    "dimension",
    "group",
    "sensor_hours",
    "sensors",
    "in_archive_share",
    "adds_up_share",
    "kept_share",
    "kept_ratio",
    "under",
    "pooled_share",
]
POOL = ["volume_tier", "day_type", "hour_local"]


def coverage(
    status: pd.DataFrame, slices: pd.DataFrame, dims: list[str], cfg: dict[str, Any]
) -> pd.DataFrame:
    """One row per (dimension, group): how much of it the archive and the profiles cover.

    `status` and `slices` are aligned row for row (one row per sensor-hour).
    """
    if status.empty:
        return pd.DataFrame(columns=COLUMNS)
    overall = float(status["kept"].mean())
    if "pooled" not in status:
        status = status.assign(pooled=False)
    flags = status[["in_archive", "adds_up", "kept", "pooled"]].astype(float)
    rows = []
    for dim in dims:
        if dim not in slices:
            continue
        g = flags.groupby(slices[dim].astype("string").fillna("unknown").to_numpy())
        sensors = status.groupby(
            slices[dim].astype("string").fillna("unknown").to_numpy()
        )["location_id"].nunique()
        for group, part in g:
            kept = float(part["kept"].mean())
            ratio = kept / overall if overall > 0 else float("nan")
            rows.append(
                {
                    "dimension": dim,
                    "group": group,
                    "sensor_hours": len(part),
                    "sensors": int(sensors[group]),
                    "in_archive_share": float(part["in_archive"].mean()),
                    "adds_up_share": float(part["adds_up"].mean()),
                    "kept_share": kept,
                    "kept_ratio": ratio,
                    "under": bool(
                        len(part) >= cfg["min_hours"] and ratio < cfg["low_ratio"]
                    ),
                    "pooled_share": float(part["pooled"].mean()),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def pooled_profiles(
    summary: pd.DataFrame, grid: pd.DataFrame | object, min_people: float
) -> pd.DataFrame:
    """Within-hour shares from the summed minutes of every trusted hour in each pooled cell.

    `summary` has location_id and the POOL columns, one row per hour of `grid` (n x 60).
    """
    cols = [*POOL, "minute", "share", "people", "n_hours", "n_sensors", "usable"]
    if summary.empty:
        return pd.DataFrame(columns=cols)
    wide = pd.concat(
        [summary[POOL].reset_index(drop=True), pd.DataFrame(grid)], axis=1
    ).dropna(subset=POOL)
    g = wide.groupby(POOL)
    sums = g[list(range(60))].sum()
    people = sums.sum(axis=1)
    cells = pd.DataFrame(
        {
            "people": people,
            "n_hours": g.size(),
            "n_sensors": summary.dropna(subset=POOL)
            .groupby(POOL)["location_id"]
            .nunique(),
        }
    )
    cells["usable"] = cells["people"] >= min_people
    shares = sums.div(people.where(people > 0), axis=0)
    long = shares.stack().rename("share").reset_index()
    long = long.rename(columns={f"level_{len(POOL)}": "minute"})
    long["minute"] = long["minute"].astype("int8")
    long["share"] = long["share"].astype("float32")
    return long.merge(cells.reset_index(), on=POOL)[cols]


def main() -> None:
    setup_logging()
    t = time.perf_counter()
    data_cfg, fcfg, bias_cfg = load_config(), load_features_config(), load_bias_config()
    cfg = bias_cfg["minute_coverage"]
    site = bias_cfg["site"]
    feats = REPO_ROOT / fcfg["processed_dir"] / "features" / site
    out = REPO_ROOT / fcfg["processed_dir"] / "bias" / site

    status = pd.read_parquet(feats / "minute_hour_status.parquet")
    if status.empty:
        table = coverage(status, pd.DataFrame(), [], cfg)
    else:
        start = status["ts_local"].min().tz_convert("UTC")
        hourly = pd.read_parquet(
            feats / "counts_hourly.parquet",
            columns=SLICE_INPUTS,
            filters=[("ts_utc", ">=", start)],
        )
        hourly = hourly.assign(location_id=hourly["location_id"].astype("int64"))
        rows = status[["location_id", "ts_local"]].merge(
            hourly, on=["location_id", "ts_local"], how="left"
        )
        afl_path = REPO_ROOT / data_cfg["raw_dir"] / "events" / "afl_attendance.csv"
        afl = (
            afl_days(pd.read_csv(afl_path), bias_cfg["afl_venues"])
            if afl_path.exists()
            else None
        )
        slices = row_slices(rows, bias_cfg, afl)
        per_sensor = pd.read_csv(out / "sensor_slices.csv")
        sensor_cols = [c for c in SENSOR_DIMS if c in per_sensor]
        slices = pd.concat(
            [
                slices.reset_index(drop=True),
                status[["location_id"]]
                .merge(
                    per_sensor[["location_id", *sensor_cols]],
                    on="location_id",
                    how="left",
                )[sensor_cols]
                .reset_index(drop=True),
            ],
            axis=1,
        )
        dims = [d for d in ROW_DIMS + SENSOR_DIMS if d in slices]

        # The mitigation: pooled profiles by volume tier, from every hour that adds up.
        minutes = mp.read_archive(
            REPO_ROOT / data_cfg["raw_dir"] / "melbourne_portal" / "minute_feed"
        )
        hours, grid = mp.minute_grid(
            minutes,
            hourly[["location_id", "ts_local", "pedestriancount"]],
            fcfg["minute_profiles"],
            which="adds_up",
        )
        tier = per_sensor.set_index("location_id")["volume_tier"]
        summary = mp.hour_summary(hours, grid)
        summary["volume_tier"] = summary["location_id"].map(tier)
        pooled = pooled_profiles(summary, grid, cfg["pooled_min_people"])
        write_parquet(pooled, feats / "minute_profiles_by_tier.parquet")
        usable = pooled.drop_duplicates(POOL).set_index(POOL)["usable"]
        cell = pd.MultiIndex.from_frame(summary[POOL])
        summary["pooled"] = usable.reindex(cell).fillna(False).to_numpy(dtype=bool)
        status = status.merge(
            summary[["location_id", "ts_local", "pooled"]],
            on=["location_id", "ts_local"],
            how="left",
        )
        status["pooled"] = status["pooled"].fillna(False).astype(bool)
        table = coverage(status.reset_index(drop=True), slices, dims, cfg)

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "minute_coverage.csv", index=False, float_format="%.4f")
    under = [f"{r.dimension}={r.group}" for r in table.itertuples() if r.under]
    write_meta(
        out / "minute_coverage_meta.json",
        config_hash=config_hash(cfg),
        sensor_hours=len(status),
        kept_share=round(float(status["kept"].mean()), 4) if len(status) else None,
        pooled_share=round(float(status["pooled"].mean()), 4)
        if "pooled" in status and len(status)
        else None,
        under_groups=under,
        runtime_s=round(time.perf_counter() - t, 1),
    )
    log.info(
        "minute coverage: %d sensor-hours, %d groups; under-covered: %s",
        len(status),
        len(table),
        under,
    )


if __name__ == "__main__":
    main()
