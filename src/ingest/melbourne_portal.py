"""Pull datasets and attachments from the City of Melbourne open data portal (Opendatasoft v2.1),
and Wayback Machine copies of portal exports the portal no longer serves.

Usage: python -m src.ingest.melbourne_portal [--only DATASET_ID ...] [--large] [--full]
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

from src.ingest.common import (
    USER_AGENT,
    download,
    load_config,
    log,
    raw_dir,
    record,
    setup_logging,
)

SOURCE = "melbourne_portal"


def dataset_meta(base_url: str, dataset_id: str) -> dict:
    r = requests.get(
        f"{base_url}/catalog/datasets/{dataset_id}",
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def wayback_url(snapshot: str, url: str) -> str:
    """Wayback Machine address of url as archived at snapshot (YYYYMMDDhhmmss).

    The id_ suffix returns the archived bytes as they were, without the Wayback page around them.
    """
    return f"https://web.archive.org/web/{snapshot}id_/{url}"


def recent_rows_filter(existing: Path, date_field: str, overlap_days: int) -> str:
    """Portal `where` clause that re-fetches only the tail of a table already held.

    Starts overlap_days before the latest date held, so the partly filled current day is
    completed and any days missed while the pipeline was off are caught up.
    """
    latest = pd.read_parquet(existing, columns=[date_field])[date_field].max()
    since = pd.Timestamp(latest).date() - timedelta(days=overlap_days)
    return f"{date_field} >= date'{since.isoformat()}'"


def merge_rolling(existing: Path, fresh: Path, key: str | list[str]) -> int:
    """Union a fresh rolling-window export into the rows already held; returns total rows.

    The portal keeps only a recent window of some tables, so overwriting would silently drop
    the oldest rows we hold. Fresh rows win on key collisions (late corrections).
    """
    old = pd.read_parquet(existing)
    new = pd.read_parquet(fresh)
    merged = (
        pd.concat([old, new], ignore_index=True)
        .drop_duplicates(subset=key, keep="last")
        .sort_values(key, ignore_index=True)
    )
    merged.to_parquet(existing, index=False)
    fresh.unlink()
    return len(merged)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", nargs="*", help="dataset ids to fetch (default: all in config)"
    )
    ap.add_argument(
        "--large", action="store_true", help="also fetch multi-GB bulk files"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="re-fetch rolling-window tables in full and re-download held attachments",
    )
    args = ap.parse_args()

    cfg = load_config()
    pc = cfg["melbourne_portal"]
    base = pc["base_url"]
    out = raw_dir(cfg, SOURCE)

    for ds, fmt in pc["datasets"].items():
        if args.only and ds not in args.only:
            continue
        meta = dataset_meta(base, ds)
        default = meta.get("metas", {}).get("default", {})
        url = f"{base}/catalog/datasets/{ds}/exports/{fmt}"
        dest = out / f"{ds}.{fmt}"
        log.info("dataset %s (%s)", ds, fmt)
        rolling = pc.get("rolling_window", {}).get(ds)
        if rolling and dest.exists():
            params = None
            if not args.full:
                where = recent_rows_filter(
                    dest, rolling["date_field"], rolling["overlap_days"]
                )
                params = {"where": where}
                log.info("fetching only %s", where)
            fresh = download(
                url, dest.with_name(f"{dest.stem}.fresh.{fmt}"), params=params
            )
            fetched = len(pd.read_parquet(fresh, columns=rolling["key"][:1]))
            rows = merge_rolling(dest, fresh, rolling["key"])
            log.info(
                "merged rolling window: %d rows fetched, %d rows held", fetched, rows
            )
        else:
            download(url, dest)
        (out / f"{ds}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        record(
            out,
            dest,
            url,
            license=default.get("license"),
            portal_modified=default.get("modified"),
            portal_records_count=default.get("records_count"),
        )

    for att in pc["attachments"]:
        if args.only and att["dataset"] not in args.only:
            continue
        kind = att.get("kind", "attachments")
        url = f"{base}/catalog/datasets/{att['dataset']}/{kind}/{att['id']}"
        dest = out / att["filename"]
        # Attachments are fixed files (the 2009-2022 counts archive, tile indexes), so a held
        # copy is kept; --full re-fetches it.
        if dest.exists() and not args.full:
            log.info(
                "attachment %s held, skipped (--full re-fetches it)", att["filename"]
            )
            continue
        log.info("attachment %s", att["filename"])
        download(url, dest)
        record(out, dest, url, dataset=att["dataset"])

    for snap in pc.get("wayback", []):
        if args.only and snap["dataset"] not in args.only:
            continue
        dest = out / snap["filename"]
        if dest.exists() and not args.full:
            log.info(
                "wayback copy %s held, skipped (--full re-fetches it)", snap["filename"]
            )
            continue
        url = wayback_url(snap["snapshot"], snap["url"])
        log.info("wayback copy %s (snapshot %s)", snap["filename"], snap["snapshot"])
        download(url, dest, timeout=1800)
        record(
            out,
            dest,
            url,
            dataset=snap["dataset"],
            original_url=snap["url"],
            snapshot=snap["snapshot"],
            license=snap["license"],
        )

    if args.large:
        for name, url in pc["large"].items():
            dest = out / url.rsplit("/", 1)[-1]
            log.info("large file %s", name)
            download(url, dest, timeout=3600)
            record(out, dest, url)


if __name__ == "__main__":
    main()
