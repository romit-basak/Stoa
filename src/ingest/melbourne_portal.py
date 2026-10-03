"""Pull datasets and attachments from the City of Melbourne open data portal (Opendatasoft v2.1).

Usage: python -m src.ingest.melbourne_portal [--only DATASET_ID ...] [--large]
"""

from __future__ import annotations

import argparse
import json

import requests

from src.ingest.common import USER_AGENT, download, load_config, raw_dir, record

SOURCE = "melbourne_portal"


def dataset_meta(base_url: str, dataset_id: str) -> dict:
    r = requests.get(
        f"{base_url}/catalog/datasets/{dataset_id}",
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", nargs="*", help="dataset ids to fetch (default: all in config)"
    )
    ap.add_argument(
        "--large", action="store_true", help="also fetch multi-GB bulk files"
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
        print(f"-> {ds} ({fmt})", flush=True)
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
        print(f"-> attachment {att['filename']}", flush=True)
        download(url, dest)
        record(out, dest, url, dataset=att["dataset"])

    if args.large:
        for name, url in pc["large"].items():
            dest = out / url.rsplit("/", 1)[-1]
            print(f"-> large {name}", flush=True)
            download(url, dest, timeout=3600)
            record(out, dest, url)


if __name__ == "__main__":
    main()
