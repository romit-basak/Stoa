"""Shared helpers for ingestion: config loading, streamed downloads, and a per-source manifest."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "Stoa/0.1 (IE7374 Northeastern student project)"


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

    Rate limits (429) and server errors (5xx) are retried with exponential backoff.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        with requests.request(
            method,
            url,
            params=params,
            data=data,
            stream=True,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        ) as r:
            if (r.status_code == 429 or r.status_code >= 500) and attempt < retries:
                wait = int(r.headers.get("Retry-After", 0)) or backoff * 2**attempt
                time.sleep(wait)
                continue
            r.raise_for_status()
            with open(tmp, "wb") as f:
                f.writelines(r.iter_content(1 << 20))
        break
    tmp.replace(dest)
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
