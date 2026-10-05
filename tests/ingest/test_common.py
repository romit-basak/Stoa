import hashlib
import json

from src.ingest import common


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"desire line" * 1000)
    assert common.sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_record_adds_and_replaces_entries(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("x\n1\n")
    common.record(tmp_path, a, "https://example.org/a", license="CC-BY-4.0")
    a.write_text("x\n2\n3\n")
    common.record(tmp_path, a, "https://example.org/a2")

    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert list(manifest) == ["a.csv"]
    entry = manifest["a.csv"]
    assert entry["url"] == "https://example.org/a2"
    assert entry["bytes"] == a.stat().st_size
    assert entry["sha256"] == common.sha256(a)
    assert "license" not in entry


def test_config_sources_are_well_formed():
    cfg = common.load_config()
    lon_min, lat_min, lon_max, lat_max = cfg["sites"]["melbourne"]["bbox"]
    assert lon_min < lon_max and lat_min < lat_max
    assert set(cfg["melbourne_portal"]["datasets"].values()) <= {
        "parquet",
        "geojson",
        "csv",
    }


class _Resp:
    def __init__(self, status, body=b"ok", headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, n):
        yield self._body


def test_download_retries_rate_limits(tmp_path, monkeypatch):
    responses = iter([_Resp(429, headers={"Retry-After": "0"}), _Resp(503), _Resp(200)])
    monkeypatch.setattr(common.requests, "request", lambda *a, **k: next(responses))
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    dest = common.download("https://example.org/x", tmp_path / "x.bin")
    assert dest.read_bytes() == b"ok"
    assert not (tmp_path / "x.bin.part").exists()
