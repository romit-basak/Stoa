"""Compare flow models trained from different start years on the same hours.

Usage: python -m src.models.flow.periods --run 2009=data/models --run 2014=DIR ...
       [--windows 2009-01-01 2022-01-01] [--output PATH]

Each run is the output directory of `train --train-start` and `select` (variants in
subdirectories). A training start limits only the training rows, so every run's out-of-fold
predictions cover the same fold rows. This scores each run's selected model and its own Space
Syntax baseline on the same windows of hours: all hours after each sensor's calibration window,
and those from each `--windows` date (Melbourne time). It also lists each run's selection
metrics: coverage, holdout precincts and worst slice. Results and reasoning are in
docs/models/training_periods.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.common import REPO_ROOT, setup_logging
from src.models.flow import data as flow_data
from src.models.flow.evaluate import cv_metrics, scored_mask

SELECTED_METRICS = (
    "coverage80_cv",
    "log_mae_holdout",
    "wape_holdout",
    "worst_slice_ratio",
)


def window_mask(ts_utc: pd.Series, start: str) -> np.ndarray:
    when = pd.Timestamp(start).tz_localize("Australia/Melbourne").tz_convert("UTC")
    if ts_utc.dt.tz is None:
        when = when.tz_localize(None)
    return (ts_utc >= when).to_numpy()


def score_run(
    label: str, root: Path, variant: str, windows: list[str], days: float
) -> list[dict]:
    out = root / variant
    summary = json.loads((out / "candidates.json").read_text())
    selected = json.loads((out / "selected.json").read_text())
    family = selected["family"]
    oof = pd.read_parquet(out / "oof.parquet")
    ts = pd.Series(oof["ts_utc"])
    base = scored_mask(oof["location_id"].to_numpy(), ts, days)
    rows = []
    for window in ["all", *windows]:
        keep = base if window == "all" else base & window_mask(ts, window)
        part = oof[keep].reset_index(drop=True)
        scores = {}
        for col in (family, "baseline_space_syntax"):
            m = cv_metrics(
                part, part["y"].to_numpy(), part[col].to_numpy(), part["fold"]
            )
            scores[col] = m
        model, ss = scores[family], scores["baseline_space_syntax"]
        rows.append(
            {
                "run": label,
                "train_start": summary.get("train_start") or "2009 (all)",
                "variant": variant,
                "family": family,
                "window": window,
                "hours": len(part),
                "log_mae_cv": model["log_mae_cv"],
                "wape_cv": model["wape_cv"],
                "space_syntax_log_mae_cv": ss["log_mae_cv"],
                "gain_vs_space_syntax": 1 - model["log_mae_cv"] / ss["log_mae_cv"],
                **{m: selected["metrics"].get(m) for m in SELECTED_METRICS},
            }
        )
    return rows


def compare(
    runs: dict[str, Path], variants: list[str], windows: list[str], days: float
) -> pd.DataFrame:
    rows = [
        r
        for label, root in runs.items()
        for v in variants
        for r in score_run(label, root, v, windows, days)
    ]
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=OUTPUT_DIR")
    ap.add_argument("--windows", nargs="*", default=["2014-01-01", "2022-01-01"])
    ap.add_argument("--variant", choices=["rich", "portable", "both"], default="both")
    ap.add_argument("--output", help="CSV path for the table")
    args = ap.parse_args()
    fcfg = flow_data.flow_config()
    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        p = Path(path)
        runs[label] = p if p.is_absolute() else REPO_ROOT / p
    variants = fcfg["variants"] if args.variant == "both" else [args.variant]
    table = compare(runs, variants, args.windows, fcfg["calibration_days"])
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
