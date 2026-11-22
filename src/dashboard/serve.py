"""Serve the generated STOA web dashboard over HTTP with range-request support for MP4 video.

Usage:
    python -m src.dashboard.serve [--dir data/dashboard] [--host 127.0.0.1] [--port 8501]
"""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "data" / "dashboard"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the STOA monitoring & simulation dashboard.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="Built dashboard directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    serve_dir = args.dir.resolve()
    if not (serve_dir / "index.html").exists():
        raise SystemExit(
            f"No index.html in {serve_dir}. Run `python -m src.dashboard.build` first."
        )

    handler = partial(SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"STOA dashboard serving {serve_dir} at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
