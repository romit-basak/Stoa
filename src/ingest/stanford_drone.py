"""Stanford Drone Dataset annotations (trajectories, no video) from a Kaggle mirror.

Usage:
    python -m src.ingest.stanford_drone [--with-reference]   # download
    python -m src.ingest.stanford_drone --parse              # tracks + counts tables

The official host (vatic2.stanford.edu) no longer serves the dataset. The Kaggle mirror
`aryashah2k/stanford-drone-dataset` is complete (8 scenes, 60 videos; its annotation files are
identical to the other full mirror's), and Kaggle serves single files of a public dataset
without logging in. Only `annotations.txt` is fetched, one per video, kept as Kaggle serves it
(zipped for the larger files, plain for the small ones);
`--with-reference` adds each video's single reference still. Videos are never fetched.

Licence: CC BY-NC-SA 3.0, the original's, whatever the mirror says (a mirror can't relicense
data). Validation and academic demonstration only: no model fitted to it may be shipped or
distributed (CLAUDE.md core rule 5), and nothing under src/features/ reads it
(tests/ingest/test_stanford_drone.py checks this).

Each annotation line is: track_id xmin ymin xmax ymax frame lost occluded generated "label",
with pixel coordinates in the scene's reference frame. `lost` marks frames where the object is
outside the view, `generated` frames interpolated by the annotation tool.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from typing import Any

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

SOURCE = "stanford_drone"
KAGGLE_API = "https://www.kaggle.com/api/v1/datasets"
COLUMNS = [
    "track_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "frame",
    "lost",
    "occluded",
    "generated",
    "label",
]


def list_files(dataset: str) -> list[dict[str, Any]]:
    """Every file in a public Kaggle dataset, following the listing's page tokens."""
    files, token = [], None
    while True:
        r = requests.get(
            f"{KAGGLE_API}/list/{dataset}",
            params={"pageToken": token} if token else None,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        r.raise_for_status()
        page = r.json()
        files += page["datasetFiles"]
        token = page.get("nextPageTokenNullable")
        if not token:
            return files


def wanted(name: str, with_reference: bool) -> bool:
    """annotations/<scene>/<video>/annotations.txt, and reference.jpg only when asked."""
    parts = name.split("/")
    if len(parts) != 4 or parts[0] != "annotations":
        return False
    return parts[3] == "annotations.txt" or (
        with_reference and parts[3] == "reference.jpg"
    )


def held(out: Path, name: str) -> Path | None:
    """The copy of a mirror file already held, plain or zipped, if any."""
    path = out / name
    for p in (path, path.with_name(path.name + ".zip")):
        if p.exists():
            return p
    return None


def keep_as_served(path: Path) -> Path:
    """Kaggle zips larger files and serves small ones as they are: name the file to match."""
    if zipfile.is_zipfile(path):
        return path.replace(path.with_name(path.name + ".zip"))
    return path


def annotation_text(path: Path) -> bytes:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return z.read("annotations.txt")
    return path.read_bytes()


def fetch(cfg: dict[str, Any], with_reference: bool) -> int:
    sd = cfg["stanford_drone"]
    out = raw_dir(cfg, SOURCE)
    files = [
        f for f in list_files(sd["kaggle_dataset"]) if wanted(f["name"], with_reference)
    ]
    log.info("%d files to hold", len(files))
    fetched = 0
    for f in files:
        if held(out, f["name"]):
            continue
        dest = out / f["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{KAGGLE_API}/download/{sd['kaggle_dataset']}"
        dest = keep_as_served(download(url, dest, params={"fileName": f["name"]}))
        # Every file is named annotations.txt, so each video folder keeps its own manifest.
        record(
            dest.parent,
            dest,
            f"{url}?fileName={f['name']}",
            mirror=sd["kaggle_dataset"],
            original=sd["original_url"],
            license=sd["license"],
            unzipped_bytes=f["totalBytes"],
        )
        fetched += 1
    log.info("downloaded %d, already held %d", fetched, len(files) - fetched)
    return fetched


def read_annotations(raw: bytes | str, scene: str, video: str) -> pd.DataFrame:
    """One video's annotation file as a table: box centre in pixels, flags as booleans."""
    text = raw.decode() if isinstance(raw, bytes) else raw
    a = pd.read_csv(io.StringIO(text), sep=" ", header=None, names=COLUMNS)
    return pd.DataFrame(
        {
            "scene": scene,
            "video": video,
            "track_id": a["track_id"].astype("int32"),
            "frame": a["frame"].astype("int32"),
            "x": ((a["xmin"] + a["xmax"]) / 2).astype("float32"),
            "y": ((a["ymin"] + a["ymax"]) / 2).astype("float32"),
            "label": a["label"].astype("category"),
            "lost": a["lost"].astype(bool),
            "occluded": a["occluded"].astype(bool),
            "generated": a["generated"].astype(bool),
        }
    )


def window_counts(tracks: pd.DataFrame, frames_per_window: int) -> pd.DataFrame:
    """Objects in view and objects first seen, per label and window of frames, per video."""
    seen = tracks[~tracks["lost"]].copy()
    seen["window"] = (seen["frame"] // frames_per_window).astype("int32")
    keys = ["scene", "video", "window", "label"]
    present = seen.groupby(keys, observed=True)["track_id"].nunique().rename("in_view")
    first = seen.groupby(["scene", "video", "track_id", "label"], observed=True)[
        "window"
    ].min()
    entering = (
        first.reset_index().groupby(keys, observed=True).size().rename("first_seen")
    )
    out = pd.concat([present, entering], axis=1).fillna(0).astype("int32")
    return out.reset_index()


def parse(cfg: dict[str, Any]) -> dict[str, int]:
    sd = cfg["stanford_drone"]
    out = raw_dir(cfg, SOURCE)
    frames = []
    for path in sorted(out.glob("annotations/*/*/annotations.txt*")):
        scene, video = path.parts[-3], path.parts[-2]
        frames.append(read_annotations(annotation_text(path), scene, video))
    if not frames:
        raise SystemExit("no annotations held: run python -m src.ingest.stanford_drone")
    tracks = pd.concat(frames, ignore_index=True)
    tracks["label"] = tracks["label"].astype("category")
    tracks.to_parquet(out / "tracks.parquet", index=False)
    counts = window_counts(tracks, sd["frames_per_window"])
    counts.to_parquet(out / "window_counts.parquet", index=False)
    stats = {
        "videos": len(frames),
        "rows": len(tracks),
        "tracks": int(tracks.groupby(["scene", "video"])["track_id"].nunique().sum()),
        "window_rows": len(counts),
    }
    log.info("parsed: %s", stats)
    return stats


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-reference", action="store_true")
    ap.add_argument("--parse", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.parse:
        parse(cfg)
    else:
        fetch(cfg, args.with_reference)


if __name__ == "__main__":
    main()
