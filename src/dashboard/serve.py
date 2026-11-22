"""Serve the generated STOA web dashboard over HTTP (local or Cloud Run via GCS sync).

Usage:
    python -m src.dashboard.serve [--dir data/dashboard] [--host 127.0.0.1] [--port 8501]
    STOA_DASHBOARD_GCS=gs://stoa-dvc-data/dashboard python -m src.dashboard.serve --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "data" / "dashboard"


def sync_from_gcs(gcs_uri: str, dest_dir: Path) -> int:
    """Download built dashboard files from a `gs://bucket/prefix` URI into `dest_dir`."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gcs_uri!r}")
    bucket_name, _, prefix = gcs_uri[5:].partition("/")
    prefix = prefix.strip("/")
    prefix_slash = f"{prefix}/" if prefix else ""

    from google.cloud import storage

    dest_dir.mkdir(parents=True, exist_ok=True)
    bucket = storage.Client().bucket(bucket_name)
    count = 0
    for blob in bucket.list_blobs(prefix=prefix_slash):
        rel = blob.name[len(prefix_slash) :] if prefix_slash else blob.name
        if not rel or rel.endswith("/"):
            continue
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target), timeout=600)
        count += 1
    print(f"synced {count} dashboard asset(s) from {gcs_uri} to {dest_dir}", flush=True)
    return count


class StoaDashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler with `/_refresh` endpoint to re-sync assets from GCS without restart."""

    gcs_source: str | None = None
    serve_root: Path = DEFAULT_DIR

    def do_GET(self) -> None:
        if self.path == "/_refresh" and self.gcs_source:
            n = sync_from_gcs(self.gcs_source, self.serve_root)
            body = f"synced {n} asset(s) from {self.gcs_source}\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the STOA monitoring & simulation dashboard."
    )
    parser.add_argument(
        "--dir", type=Path, default=DEFAULT_DIR, help="Built dashboard directory"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8501")))
    parser.add_argument(
        "--gcs-source",
        default=os.environ.get("STOA_DASHBOARD_GCS", ""),
        help="Optional gs://bucket/prefix URI to sync dashboard assets from on startup",
    )
    args = parser.parse_args()

    serve_dir = args.dir.resolve()
    gcs_source = args.gcs_source or None
    if gcs_source:
        sync_from_gcs(gcs_source, serve_dir)

    if not (serve_dir / "index.html").exists():
        raise SystemExit(
            f"No index.html in {serve_dir}. Run `python -m src.dashboard.build` first."
        )

    StoaDashboardHandler.gcs_source = gcs_source
    StoaDashboardHandler.serve_root = serve_dir
    handler = partial(StoaDashboardHandler, directory=str(serve_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"STOA dashboard serving {serve_dir} at http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
