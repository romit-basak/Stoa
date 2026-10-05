"""NYC DOT pedestrian counts from NYC Open Data (Socrata), for the cross-city transfer test only.

Usage: python -m src.ingest.nyc

Evaluation data: never feed these files into training (CLAUDE.md "Generalization").
See docs/data_cards/nyc_dot_pedestrian_counts.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ingest.common import download, load_config, raw_dir, record

SOURCE = "nyc"
LICENSE = "NYC Open Data terms (no licence on dataset page; verify). Evaluation only."


def fetch_paged_csv(
    url: str, dest: Path, params: dict[str, Any], page_size: int
) -> int:
    """Page through a Socrata CSV endpoint into one file. Returns the number of data rows."""
    rows = 0
    with open(dest.with_suffix(".csv.part"), "w", encoding="utf-8") as out:
        offset = 0
        while True:
            page = dest.with_name(f"{dest.stem}.page")
            download(
                url, page, params={**params, "$limit": page_size, "$offset": offset}
            )
            lines = page.read_text(encoding="utf-8").splitlines(keepends=True)
            page.unlink()
            if offset == 0:
                out.write(lines[0])
            body = lines[1:]
            out.writelines(body)
            rows += len(body)
            if len(body) < page_size:
                break
            offset += page_size
    dest.with_suffix(".csv.part").replace(dest)
    return rows


def main() -> None:
    cfg = load_config()
    nyc = cfg["nyc_open_data"]
    out = raw_dir(cfg, SOURCE)

    for name, ds in nyc["datasets"].items():
        url = f"{nyc['base_url']}/{ds['id']}.{ds['format']}"
        dest = out / f"{name}.{ds['format']}"
        params: dict[str, Any] = {}
        if "where" in ds:
            params["$where"] = ds["where"]
        if ds["format"] == "csv":
            params["$order"] = ds.get("order", ":id")
            n = fetch_paged_csv(url, dest, params, nyc["page_size"])
            record(out, dest, url, params=params, rows=n, license=LICENSE)
        else:
            params["$limit"] = nyc["page_size"]
            download(url, dest, params=params)
            record(out, dest, url, params=params, license=LICENSE)

    readme = out / "bi-annual-ped-count-readme.pdf"
    download(nyc["methodology"], readme)
    record(out, readme, nyc["methodology"])


if __name__ == "__main__":
    main()
