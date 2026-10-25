"""Archive the City of Melbourne's per-minute pedestrian counts, which the portal doesn't keep.

Usage:
    python -m src.ingest.minute_feed [--archive-dir DIR] [--changed-list FILE]
    python -m src.ingest.minute_feed --sync [--archive-dir DIR]

The portal's "past hour (counts per minute)" dataset holds only the latest stretch of minutes
and is refreshed every 15 minutes, so anything not saved is gone. A scheduled GitHub Actions
workflow (.github/workflows/minute_feed.yml) polls it every 15 minutes, merges each poll into
one file per local day and keeps the files as assets of a GitHub Release, one release a year.
--sync copies those assets into data/raw/melbourne_portal/minute_feed/, where DVC versions them.

The archive keeps rows as the portal published them. Minutes with no one counted are absent,
not zero, and the portal sometimes publishes two different rows for one sensor and minute
(its description notes this for sensors 67-69), so only rows identical in every column are
merged; `polled_at` is the first poll that saw each row.
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.ingest.common import (
    REPO_ROOT,
    USER_AGENT,
    download,
    load_config,
    log,
    record,
    setup_logging,
)

COLUMNS = [
    "location_id",
    "sensing_datetime",
    "sensing_date",
    "sensing_time",
    "direction_1",
    "direction_2",
    "total_of_directions",
]


def day_file(archive: Path, day: str) -> Path:
    return archive / f"minute_feed_{day}.parquet"


def normalise(poll: pd.DataFrame, polled_at: datetime) -> pd.DataFrame:
    """The export's columns with fixed types, plus the poll time."""
    missing = [c for c in COLUMNS if c not in poll.columns]
    if missing:
        raise ValueError(f"the export has no {missing} column(s)")
    out = poll[COLUMNS].copy()
    out["location_id"] = out["location_id"].astype("int32")
    out["sensing_datetime"] = pd.to_datetime(
        out["sensing_datetime"], utc=True
    ).dt.as_unit("s")
    out["sensing_date"] = out["sensing_date"].astype(str)
    out["sensing_time"] = out["sensing_time"].astype(str)
    for c in ["direction_1", "direction_2", "total_of_directions"]:
        out[c] = out[c].astype("int32")
    out["polled_at"] = pd.Timestamp(polled_at).tz_convert("UTC").as_unit("s")
    return out


def merge(held: pd.DataFrame | None, poll: pd.DataFrame) -> pd.DataFrame:
    """Union of the rows held and a new poll; identical rows keep their earliest poll time."""
    both = poll if held is None else pd.concat([held, poll], ignore_index=True)
    both = both.sort_values("polled_at", kind="stable")
    out = both.drop_duplicates(COLUMNS, keep="first")
    return out.sort_values(
        ["sensing_datetime", "location_id", "polled_at"], ignore_index=True
    )


def add_poll(archive: Path, poll: pd.DataFrame) -> list[Path]:
    """Merge a normalised poll into the day files; returns the files that changed."""
    archive.mkdir(parents=True, exist_ok=True)
    changed = []
    for day, rows in poll.groupby("sensing_date", sort=True):
        path = day_file(archive, str(day))
        held = pd.read_parquet(path) if path.exists() else None
        merged = merge(held, rows)
        if held is not None and len(merged) == len(held):
            continue
        merged.to_parquet(path, index=False)
        log.info(
            "%s: %d rows (+%d)",
            path.name,
            len(merged),
            len(merged) - (0 if held is None else len(held)),
        )
        changed.append(path)
    return changed


def poll_once(cfg: dict[str, Any], archive: Path) -> list[Path]:
    mf = cfg["minute_feed"]
    url = f"{cfg['melbourne_portal']['base_url']}/catalog/datasets/{mf['dataset']}/exports/parquet"
    with tempfile.TemporaryDirectory() as tmp:
        dest = download(url, Path(tmp) / "poll.parquet", timeout=120)
        polled_at = datetime.now(UTC)
        poll = normalise(pd.read_parquet(dest), polled_at)
    log.info(
        "poll: %d rows, %d sensors, %s to %s",
        len(poll),
        poll["location_id"].nunique(),
        poll["sensing_datetime"].min(),
        poll["sensing_datetime"].max(),
    )
    changed = add_poll(archive, poll)
    for path in changed:
        record(archive, path, url, dataset=mf["dataset"], license=mf["license"])
    return changed


def release_assets(repo: str, prefix: str) -> list[dict[str, Any]]:
    """Day-file assets of the repo's archive releases (public repo: no token needed)."""
    r = requests.get(
        f"https://api.github.com/repos/{repo}/releases",
        params={"per_page": 100},
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
        timeout=60,
    )
    r.raise_for_status()
    return [
        a
        for rel in r.json()
        if rel["tag_name"].startswith(prefix)
        for a in rel["assets"]
        if a["name"].startswith("minute_feed_") and a["name"].endswith(".parquet")
    ]


def sync(cfg: dict[str, Any], archive: Path) -> int:
    """Download every archived day file that is missing here or differs in size."""
    mf = cfg["minute_feed"]
    archive.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for asset in release_assets(mf["github_repo"], mf["release_prefix"]):
        dest = archive / asset["name"]
        if dest.exists() and dest.stat().st_size == asset["size"]:
            continue
        download(asset["browser_download_url"], dest)
        record(
            archive,
            dest,
            asset["browser_download_url"],
            dataset=mf["dataset"],
            license=mf["license"],
            updated_at=asset["updated_at"],
        )
        fetched += 1
    log.info("synced %d day file(s) into %s", fetched, archive)
    return fetched


def main() -> None:
    setup_logging()
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archive-dir",
        type=Path,
        default=REPO_ROOT / cfg["raw_dir"] / "melbourne_portal" / "minute_feed",
    )
    ap.add_argument(
        "--changed-list", type=Path, help="write the changed day files here"
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help="download the archived day files from the GitHub Releases instead of polling",
    )
    args = ap.parse_args()
    if args.sync:
        sync(cfg, args.archive_dir)
        return
    changed = poll_once(cfg, args.archive_dir)
    if args.changed_list:
        args.changed_list.write_text("".join(f"{p}\n" for p in changed))


if __name__ == "__main__":
    main()
