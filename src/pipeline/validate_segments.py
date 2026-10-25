"""Checks on a site's segment features before they are versioned.

Usage: python -m src.pipeline.validate_segments --site SITE

The hourly-counts validation (src/pipeline/validate.py) covers Melbourne's training table;
this covers the per-segment table every site builds (osm_segment_features.parquet), which is
all a demonstration or transfer-test site has. Errors fail the task, as in validate:
- schema: the columns and types declared in src/features/schema.py;
- keys: segment_id is unique;
- metrics: betweenness and reach are present for every segment, and reach never shrinks as
  the radius grows (a bug in the network code shows up here first);
- ranges: lengths positive, coverage a share, counts and distances non-negative and no
  farther than the configured cap.
The main component's share of segment length is a warning: a low share means the walk filter
or the extract cut the network into pieces, which is worth a look but not always wrong.

Writes data/processed/validation/<site>/segments_validation.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from typing import Any

import geopandas as gpd
import pandas as pd

from src.features import schema
from src.features.common import load_features_config, site_settings
from src.ingest.common import REPO_ROOT, load_config, setup_logging
from src.pipeline.validate import (
    ERROR,
    WARNING,
    Check,
    ValidationError,
    check_schema,
    log,
)

MIN_MAIN_COMPONENT_SHARE = 0.8


def check_metrics(df: pd.DataFrame, radii: list[int]) -> list[Check]:
    checks = []
    present = [f"met_betweenness_{r}" for r in radii] + [
        f"met_reach_{r}" for r in radii
    ]
    missing = {
        c: int(df[c].isna().sum()) for c in present if c in df and df[c].isna().any()
    }
    checks.append(
        Check(
            "metrics present",
            ERROR,
            not missing,
            ", ".join(f"{c}: {n} missing" for c, n in missing.items()),
            missing,
        )
    )
    shrinking = {}
    for a, b in itertools.pairwise(radii):
        if f"met_reach_{a}" in df and f"met_reach_{b}" in df:
            n = int((df[f"met_reach_{b}"] < df[f"met_reach_{a}"]).sum())
            if n:
                shrinking[f"{a}->{b}"] = n
    checks.append(
        Check(
            "reach grows with radius",
            ERROR,
            not shrinking,
            ", ".join(f"{k}: {n} segments" for k, n in shrinking.items()),
            shrinking,
        )
    )
    return checks


def check_ranges(df: pd.DataFrame, cap_m: float) -> list[Check]:
    rules: dict[str, tuple[float | None, float | None]] = {
        "length_m": (1e-6, None),
        "building_coverage": (0.0, 1.0),
        "building_count": (0.0, None),
        "degree_min": (1.0, None),
    }
    for c in df.columns:
        if c.startswith("poi_"):
            rules[c] = (0.0, None)
        if c.endswith("_dist_m"):
            rules[c] = (0.0, cap_m)
    bad = {}
    for c, (lo, hi) in rules.items():
        if c not in df:
            continue
        v = df[c].dropna().astype(float)
        n = int((v < lo).sum()) if lo is not None else 0
        n += int((v > hi).sum()) if hi is not None else 0
        if n:
            bad[c] = n
    return [
        Check(
            "ranges",
            ERROR,
            not bad,
            ", ".join(f"{c}: {n} out of range" for c, n in bad.items()),
            bad,
        )
    ]


def check_connectivity(df: pd.DataFrame) -> Check:
    share = float(
        df.loc[df["in_main_component"], "length_m"].sum() / df["length_m"].sum()
    )
    return Check(
        "main component share of length",
        WARNING,
        share >= MIN_MAIN_COMPONENT_SHARE,
        f"{share:.1%} of segment length is in the largest connected part",
        {"share": round(share, 4)},
    )


def validate_segments(df: pd.DataFrame, fcfg: dict[str, Any]) -> list[Check]:
    cols = schema.segment_features(fcfg)
    expected = {"columns": {c.name: c.dtype for c in cols}}
    return [
        *check_schema(df, expected),
        Check(
            "keys: segment_id unique",
            ERROR,
            bool(df["segment_id"].is_unique),
            f"{int(df['segment_id'].duplicated().sum())} duplicated",
        ),
        *check_metrics(df, fcfg["network_metrics"]["radii_m"]),
        *check_ranges(df, fcfg["built_form"]["max_distance_m"]),
        check_connectivity(df),
    ]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    args = ap.parse_args()
    data_cfg, fcfg = load_config(), load_features_config()
    site = site_settings(args.site, data_cfg, fcfg)
    path = site.processed / "osm_segment_features.parquet"
    df = gpd.read_parquet(path)
    log.info("read %s: %d segments", path.relative_to(REPO_ROOT), len(df))
    checks = validate_segments(df, fcfg)
    for c in checks:
        status = "PASS" if c.passed else ("FAIL" if c.severity == ERROR else "WARN")
        log.info("%s  %s%s", status, c.name, f": {c.detail}" if c.detail else "")
    out = REPO_ROOT / fcfg["processed_dir"] / "validation" / site.name
    out.mkdir(parents=True, exist_ok=True)
    errors = [c for c in checks if not c.passed and c.severity == ERROR]
    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "site": site.name,
        "segments": len(df),
        "passed": not errors,
        "checks": [c.__dict__ for c in checks],
    }
    (out / "segments_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise ValidationError(f"{len(errors)} error check(s) failed for {site.name}")


if __name__ == "__main__":
    main()
