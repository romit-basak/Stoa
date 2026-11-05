import io
import zipfile

import pytest

from src.ingest import stanford_drone as sd
from src.ingest.common import REPO_ROOT

# The fifth line is "lost": the biker is outside the view in that frame.
SAMPLE = """\
0 213 1038 241 1072 0 0 0 0 "Pedestrian"
0 215 1040 243 1074 1 0 0 1 "Pedestrian"
0 215 1040 243 1074 1800 0 0 0 "Pedestrian"
1 10 20 30 40 5 0 1 0 "Biker"
1 12 22 32 42 6 1 0 0 "Biker"
2 50 50 60 60 1805 0 0 0 "Pedestrian"
"""


def test_only_annotation_files_are_wanted():
    assert sd.wanted("annotations/bookstore/video0/annotations.txt", False)
    assert not sd.wanted("annotations/bookstore/video0/reference.jpg", False)
    assert sd.wanted("annotations/bookstore/video0/reference.jpg", True)
    assert not sd.wanted("video/bookstore/video0/video.mp4", True)
    assert not sd.wanted("annotations/readme.txt", True)


def test_files_are_named_by_how_kaggle_served_them(tmp_path):
    plain = tmp_path / "a" / "annotations.txt"
    plain.parent.mkdir()
    plain.write_text(SAMPLE)
    assert sd.keep_as_served(plain) == plain
    zipped = tmp_path / "b" / "annotations.txt"
    zipped.parent.mkdir()
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("annotations.txt", SAMPLE)
    kept = sd.keep_as_served(zipped)
    assert kept.name == "annotations.txt.zip"
    assert sd.annotation_text(kept) == sd.annotation_text(plain)
    assert sd.held(tmp_path, "b/annotations.txt") == kept
    assert sd.held(tmp_path, "c/annotations.txt") is None


def test_annotations_parse_to_box_centres_and_flags():
    t = sd.read_annotations(SAMPLE, "bookstore", "video0")
    assert len(t) == 6
    first = t.iloc[0]
    assert (first["x"], first["y"]) == (227.0, 1055.0)
    assert first["label"] == "Pedestrian"
    assert t["generated"].tolist() == [False, True, False, False, False, False]
    assert t["lost"].sum() == 1
    assert set(t["scene"]) == {"bookstore"}


def test_counts_per_window_skip_lost_frames():
    t = sd.read_annotations(SAMPLE, "bookstore", "video0")
    c = sd.window_counts(t, frames_per_window=1800).set_index(["window", "label"])
    # Window 0: pedestrian 0 and biker 1 in view, both first seen there.
    assert c.loc[(0, "Pedestrian"), "in_view"] == 1
    assert c.loc[(0, "Biker"), "in_view"] == 1
    # Window 1: pedestrian 0 still in view (not new), pedestrian 2 new.
    assert c.loc[(1, "Pedestrian"), "in_view"] == 2
    assert c.loc[(1, "Pedestrian"), "first_seen"] == 1


def test_parse_reads_the_held_zips(tmp_path, monkeypatch):
    d = tmp_path / "annotations" / "quad" / "video3"
    d.mkdir(parents=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("annotations.txt", SAMPLE)
    (d / "annotations.txt.zip").write_bytes(buf.getvalue())
    monkeypatch.setattr(sd, "raw_dir", lambda cfg, source: tmp_path)
    stats = sd.parse({"stanford_drone": {"frames_per_window": 1800}})
    assert stats["videos"] == 1 and stats["tracks"] == 3
    assert (tmp_path / "tracks.parquet").exists()


def test_parse_fails_clearly_with_nothing_held(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "raw_dir", lambda cfg, source: tmp_path)
    with pytest.raises(SystemExit, match="no annotations held"):
        sd.parse({"stanford_drone": {"frames_per_window": 1800}})


def test_no_feature_code_reads_the_drone_data():
    # Core rule 5: the dataset is for validation and demonstration only.
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "src" / "features").rglob("*.py")
        if "stanford_drone" in p.read_text()
    ]
    assert offenders == []
