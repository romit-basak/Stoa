"""Shared helpers for ingestion: config loading, streamed downloads, and a per-source manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "Stoa/0.1 (IE7374 Northeastern student project)"

log = logging.getLogger("stoa.ingest")


def setup_logging(level: int = logging.INFO) -> None:
    """Log to stderr with timestamps; Airflow captures it in each task's log."""
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or REPO_ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def raw_dir(cfg: dict[str, Any], source: str) -> Path:
    d = REPO_ROOT / cfg["raw_dir"] / source
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DownloadError(RuntimeError):
    """A download that failed after every retry, or came back empty or cut short."""


def _write_body(r: requests.Response, tmp: Path) -> None:
    """Write the response to tmp; raise if it is empty or shorter than the server said."""
    with open(tmp, "wb") as f:
        f.writelines(r.iter_content(1 << 20))
    size = tmp.stat().st_size
    if size == 0:
        raise DownloadError(f"{r.url} returned an empty body")
    # Content-Length counts the bytes on the wire; with gzip those differ from what we save.
    expected = r.headers.get("Content-Length")
    plain = r.headers.get("Content-Encoding", "identity") == "identity"
    if expected and plain and size != int(expected):
        raise DownloadError(f"{r.url} was cut short: got {size} of {expected} bytes")


def download(
    url: str,
    dest: Path,
    params: dict[str, Any] | None = None,
    method: str = "GET",
    data: Any = None,
    timeout: int = 600,
    retries: int = 5,
    backoff: float = 10.0,
) -> Path:
    """Stream url to dest via a .part file so an interrupted download never looks complete.

    Rate limits (429), server errors (5xx), dropped connections, timeouts and empty or
    truncated bodies are retried with exponential backoff. Other HTTP errors (a 404, say)
    fail at once. If every try fails, raise DownloadError and leave any existing dest as it
    was.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s -> %s", url, dest.name)
    for attempt in range(retries + 1):
        wait = backoff * 2**attempt
        try:
            with requests.request(
                method,
                url,
                params=params,
                data=data,
                stream=True,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            ) as r:
                if r.status_code == 429 or r.status_code >= 500:
                    wait = int(r.headers.get("Retry-After", 0)) or wait
                    problem = f"HTTP {r.status_code}"
                else:
                    r.raise_for_status()
                    _write_body(r, tmp)
                    break
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,  # connection dropped mid-body
            DownloadError,
        ) as e:
            problem = str(e)
        tmp.unlink(missing_ok=True)
        if attempt == retries:
            log.error("giving up on %s after %d tries: %s", url, retries + 1, problem)
            raise DownloadError(f"{url} failed after {retries + 1} tries: {problem}")
        log.warning(
            "%s from %s, retry %d/%d in %.0f s",
            problem,
            url,
            attempt + 1,
            retries,
            wait,
        )
        time.sleep(wait)
    tmp.replace(dest)
    log.info("saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def record(source_dir: Path, path: Path, url: str, **extra: Any) -> None:
    """Add/replace an entry for path in source_dir/MANIFEST.json."""
    manifest_path = source_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[path.name] = {
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
