"""Archive the City of Melbourne's per-minute pedestrian counts, which the portal doesn't keep.

Usage:
    python -m src.ingest.minute_feed [--bucket | --dest gs://BUCKET/PREFIX | --dest DIR]
    python -m src.ingest.minute_feed --sync [--archive-dir DIR]

The portal's "past hour (counts per minute)" dataset is refreshed every 15 minutes and holds counts
from 00:55 Melbourne time until a reset at about 02:00 the next morning, so anything not saved by
then is gone. The
`stoa_minute_feed` DAG polls it every 15 minutes and merges each poll into one parquet file per
Melbourne day in the project bucket (--bucket: `minute_feed.gcs` in configs/data.yaml). Without
--bucket or --dest the day files go to data/raw/melbourne_portal/minute_feed/. --sync copies the
bucket's day files there, where DVC versions them with the rest of data/raw.

Several machines can poll at once: each write uploads only if the day file is unchanged since it
was read (a generation precondition), and otherwise reads it again and retries, so no machine
overwrites another's rows.

The archive keeps rows as the portal published them. Minutes with no one counted are absent,
not zero, and the portal sometimes publishes two different rows for one sensor and minute
(its description notes this for sensors 67-69), so only rows identical in every column are
merged; `polled_at` is the first poll that saw each row.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from src.ingest.common import (
    REPO_ROOT,
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
GCS_API = "https://storage.googleapis.com/storage/v1"
GCS_UPLOAD = "https://storage.googleapis.com/upload/storage/v1"
# @task.bash marks a task skipped (not failed) when its command exits with 99.
SKIP_EXIT_CODE = 99


def day_name(day: str) -> str:
    return f"minute_feed_{day}.parquet"


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


def to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


# --------------------------------------------------------------------------------------------
# Where the day files live: a local folder, or a bucket prefix
# --------------------------------------------------------------------------------------------


class Conflict(RuntimeError):
    """Someone else wrote the day file between our read and our write."""


class LocalStore:
    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def read(self, name: str) -> tuple[bytes | None, Any]:
        path = self.folder / name
        if not path.exists():
            return None, None
        return path.read_bytes(), path.stat().st_mtime_ns

    def write(self, name: str, data: bytes, token: Any) -> None:
        path = self.folder / name
        now = path.stat().st_mtime_ns if path.exists() else None
        if now != token:
            raise Conflict(name)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)

    def __str__(self) -> str:
        return str(self.folder)


class GcsStore:
    """Objects under gs://bucket/prefix, through the JSON API with generation preconditions."""

    def __init__(self, url: str, session: Any):
        bucket, _, prefix = url.removeprefix("gs://").partition("/")
        self.bucket, self.prefix, self.session = bucket, prefix.strip("/"), session

    def _object(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def read(self, name: str) -> tuple[bytes | None, Any]:
        obj = quote(self._object(name), safe="")
        r = self.session.get(
            f"{GCS_API}/b/{self.bucket}/o/{obj}", params={"alt": "media"}, timeout=60
        )
        if r.status_code == 404:
            return None, 0  # generation 0: "only if it doesn't exist yet"
        r.raise_for_status()
        return r.content, int(r.headers["x-goog-generation"])

    def write(self, name: str, data: bytes, token: Any) -> None:
        r = self.session.post(
            f"{GCS_UPLOAD}/b/{self.bucket}/o",
            params={
                "uploadType": "media",
                "name": self._object(name),
                "ifGenerationMatch": token,
            },
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=120,
        )
        if r.status_code == 412:
            raise Conflict(name)
        r.raise_for_status()

    def list(self) -> list[dict[str, Any]]:
        items, token = [], None
        while True:
            params = {"prefix": f"{self.prefix}/" if self.prefix else ""}
            if token:
                params["pageToken"] = token
            r = self.session.get(
                f"{GCS_API}/b/{self.bucket}/o", params=params, timeout=60
            )
            r.raise_for_status()
            page = r.json()
            items += page.get("items", [])
            token = page.get("nextPageToken")
            if not token:
                return items

    def __str__(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}"


def gcs_session() -> Any:
    """An authorised session for the bucket, found the way dvc_push finds credentials."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    from src.pipeline.dvc_push import check_credentials

    check_credentials()
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/devstorage.read_write"]
    )
    return AuthorizedSession(creds)


def merge_into(store: Any, name: str, rows: pd.DataFrame, attempts: int = 5) -> int:
    """Merge rows into one day file; returns rows added (0 means nothing was written)."""
    for _ in range(attempts):
        data, token = store.read(name)
        held = pd.read_parquet(io.BytesIO(data)) if data is not None else None
        merged = merge(held, rows)
        added = len(merged) - (0 if held is None else len(held))
        if added == 0:
            return 0
        try:
            store.write(name, to_bytes(merged), token)
            return added
        except Conflict:
            log.info("%s changed while merging, reading it again", name)
    raise RuntimeError(f"{name}: still changing after {attempts} attempts")


def add_poll(store: Any, poll: pd.DataFrame) -> dict[str, int]:
    """Merge a normalised poll into the day files; returns rows added per changed file."""
    added = {}
    for day, rows in poll.groupby("sensing_date", sort=True):
        n = merge_into(store, day_name(str(day)), rows)
        if n:
            log.info("%s/%s: +%d rows", store, day_name(str(day)), n)
            added[day_name(str(day))] = n
    return added


def poll_since(now: datetime, recent_hours: float) -> datetime | None:
    """Where a poll starts: the whole feed in each hour's first quarter, else the last hours.

    The portal takes about 8 s to export the whole day but under a second for the last few
    hours, and a poll only needs minutes it may not have seen. The hourly full export picks up
    any minute the portal published late, or that every poll in a longer gap missed.
    """
    if now.minute < 15:
        return None
    return (now - timedelta(hours=recent_hours)).replace(second=0, microsecond=0)


def fetch_poll(cfg: dict[str, Any], since: datetime | None = None) -> pd.DataFrame:
    """The feed's rows (only those from `since` on, if given) with fixed types."""
    mf = cfg["minute_feed"]
    url = f"{cfg['melbourne_portal']['base_url']}/catalog/datasets/{mf['dataset']}/exports/parquet"
    params = (
        None
        if since is None
        else {"where": f"sensing_datetime >= date'{since.isoformat()}'"}
    )
    with tempfile.TemporaryDirectory() as tmp:
        dest = download(url, Path(tmp) / "poll.parquet", params=params, timeout=120)
        poll = normalise(pd.read_parquet(dest), datetime.now(UTC))
    log.info(
        "poll %s", "of the whole feed" if since is None else f"from {since.isoformat()}"
    )
    log.info(
        "poll: %d rows, %d sensors, %s to %s",
        len(poll),
        poll["location_id"].nunique(),
        poll["sensing_datetime"].min(),
        poll["sensing_datetime"].max(),
    )
    return poll


def sync(store: GcsStore, folder: Path, cfg: dict[str, Any]) -> int:
    """Copy every day file that is missing locally or differs in size."""
    folder.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for item in store.list():
        name = item["name"].rsplit("/", 1)[-1]
        if not (name.startswith("minute_feed_") and name.endswith(".parquet")):
            continue
        dest = folder / name
        if dest.exists() and dest.stat().st_size == int(item["size"]):
            continue
        data, _ = store.read(name)
        dest.write_bytes(data)
        record(
            folder,
            dest,
            f"gs://{store.bucket}/{item['name']}",
            dataset=cfg["minute_feed"]["dataset"],
            license=cfg["minute_feed"]["license"],
            generation=item["generation"],
        )
        fetched += 1
    log.info("synced %d day file(s) from %s into %s", fetched, store, folder)
    return fetched


def main() -> None:
    setup_logging()
    cfg = load_config()
    local = REPO_ROOT / cfg["raw_dir"] / "melbourne_portal" / "minute_feed"
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bucket",
        action="store_true",
        help="write to minute_feed.gcs in configs/data.yaml",
    )
    ap.add_argument(
        "--dest", help="gs://bucket/prefix or a folder (default: data/raw/...)"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="fetch the whole feed, not only the last hours",
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help="copy the bucket's day files into data/raw instead of polling",
    )
    args = ap.parse_args()
    dest = cfg["minute_feed"]["gcs"] if args.bucket else (args.dest or str(local))
    from src.pipeline.dvc_push import MissingCredentials

    try:
        if args.sync:
            sync(GcsStore(cfg["minute_feed"]["gcs"], gcs_session()), local, cfg)
            return
        store = (
            GcsStore(dest, gcs_session())
            if dest.startswith("gs://")
            else LocalStore(Path(dest))
        )
        now = datetime.now(UTC)
        since = (
            None if args.full else poll_since(now, cfg["minute_feed"]["recent_hours"])
        )
        add_poll(store, fetch_poll(cfg, since))
    except MissingCredentials as err:
        # A clone without bucket access: skip (pink in Airflow) instead of failing every poll.
        log.warning("%s; skipped", err)
        sys.exit(SKIP_EXIT_CODE)


if __name__ == "__main__":
    main()
