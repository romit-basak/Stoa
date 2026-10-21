import hashlib
import json

import pytest
import requests

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


class FakeResponse:
    """Stands in for a streamed requests.Response in download tests."""

    def __init__(self, status=200, body=b"data", headers=None):
        self.status_code = status
        self.body = body
        self.headers = (
            headers if headers is not None else {"Content-Length": str(len(body))}
        )
        self.url = "https://example.org/f"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self.body


def fake_server(monkeypatch, *replies):
    """Make requests.request return (or raise) each reply in turn; return the call log."""
    calls = []
    replies = iter(replies)

    def request(*args, **kwargs):
        calls.append(args)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(common.requests, "request", request)
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    return calls


def test_download_saves_body(tmp_path, monkeypatch):
    fake_server(monkeypatch, FakeResponse(body=b"hello"))
    dest = common.download("https://example.org/f", tmp_path / "f.csv")
    assert dest.read_bytes() == b"hello"
    assert not (tmp_path / "f.csv.part").exists()


def test_download_retries_server_errors_and_dropped_connections(tmp_path, monkeypatch):
    calls = fake_server(
        monkeypatch,
        FakeResponse(status=503),
        requests.ConnectionError("reset"),
        requests.Timeout("slow"),
        FakeResponse(body=b"ok"),
    )
    dest = common.download("https://example.org/f", tmp_path / "f.csv")
    assert dest.read_bytes() == b"ok"
    assert len(calls) == 4


def test_download_retries_truncated_and_empty_bodies(tmp_path, monkeypatch):
    calls = fake_server(
        monkeypatch,
        FakeResponse(body=b"hal", headers={"Content-Length": "5"}),
        FakeResponse(body=b""),
        FakeResponse(body=b"whole"),
    )
    dest = common.download("https://example.org/f", tmp_path / "f.csv")
    assert dest.read_bytes() == b"whole"
    assert len(calls) == 3


def test_download_ignores_length_of_compressed_body(tmp_path, monkeypatch):
    # With gzip, Content-Length is the compressed size, not what iter_content yields.
    headers = {"Content-Length": "3", "Content-Encoding": "gzip"}
    fake_server(monkeypatch, FakeResponse(body=b"unzipped", headers=headers))
    dest = common.download("https://example.org/f", tmp_path / "f.csv")
    assert dest.read_bytes() == b"unzipped"


def test_download_gives_up_and_keeps_the_old_file(tmp_path, monkeypatch):
    dest = tmp_path / "f.csv"
    dest.write_bytes(b"last good copy")
    fake_server(monkeypatch, *[FakeResponse(status=500)] * 3)
    with pytest.raises(common.DownloadError, match="after 3 tries"):
        common.download("https://example.org/f", dest, retries=2)
    assert dest.read_bytes() == b"last good copy"
    assert not (tmp_path / "f.csv.part").exists()


def test_download_fails_fast_on_client_errors(tmp_path, monkeypatch):
    calls = fake_server(monkeypatch, FakeResponse(status=404))
    with pytest.raises(requests.HTTPError):
        common.download("https://example.org/f", tmp_path / "f.csv")
    assert len(calls) == 1
