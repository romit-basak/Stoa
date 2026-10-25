import io
from datetime import UTC, datetime

import pandas as pd
import pytest

from src.ingest import minute_feed as mf

T0 = datetime(2026, 10, 25, 5, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 25, 5, 15, tzinfo=UTC)
DAY = "minute_feed_2026-10-25.parquet"


def export(*rows):
    """Rows as the portal's parquet export has them: (sensor, UTC minute, local date, d1, d2)."""
    return pd.DataFrame(
        [
            {
                "location_id": lid,
                "sensing_datetime": pd.Timestamp(ts, tz="UTC"),
                "sensing_date": day,
                "sensing_time": pd.Timestamp(ts, tz="UTC")
                .tz_convert("Australia/Melbourne")
                .strftime("%H:%M"),
                "direction_1": d1,
                "direction_2": d2,
                "total_of_directions": d1 + d2,
            }
            for lid, ts, day, d1, d2 in rows
        ]
    )


def poll(at, *rows):
    return mf.normalise(export(*rows), at)


def read_day(store, name=DAY):
    data, _ = store.read(name)
    return pd.read_parquet(io.BytesIO(data))


def test_normalise_fixes_types_and_adds_the_poll_time():
    out = poll(T0, (61, "2026-10-25 05:04", "2026-10-25", 2, 4))
    assert out.columns.tolist() == mf.COLUMNS + ["polled_at"]
    assert str(out["location_id"].dtype) == "int32"
    assert out.loc[0, "sensing_time"] == "16:04"  # local clock, as published
    assert out.loc[0, "polled_at"] == pd.Timestamp(T0)


def test_normalise_rejects_a_changed_export():
    rows = export((61, "2026-10-25 05:04", "2026-10-25", 2, 4))
    with pytest.raises(ValueError, match="total_of_directions"):
        mf.normalise(rows.drop(columns="total_of_directions"), T0)


def test_overlapping_polls_keep_one_row_with_the_first_poll_time(tmp_path):
    store = mf.LocalStore(tmp_path)
    a = poll(
        T0,
        (61, "2026-10-25 05:00", "2026-10-25", 1, 1),
        (61, "2026-10-25 05:01", "2026-10-25", 2, 0),
    )
    b = poll(
        T1,
        (61, "2026-10-25 05:01", "2026-10-25", 2, 0),  # seen before
        (61, "2026-10-25 05:02", "2026-10-25", 0, 3),  # new
    )
    assert mf.add_poll(store, a) == {DAY: 2}
    assert mf.add_poll(store, b) == {DAY: 1}
    out = read_day(store)
    assert len(out) == 3
    seen_twice = out[
        out["sensing_datetime"] == pd.Timestamp("2026-10-25 05:01", tz="UTC")
    ]
    assert seen_twice["polled_at"].tolist() == [pd.Timestamp(T0)]
    # Repeating a poll writes nothing.
    assert mf.add_poll(store, b) == {}


def test_two_different_rows_for_one_minute_are_both_kept(tmp_path):
    # The portal sometimes publishes two readings for a sensor and minute; keep what it said.
    store = mf.LocalStore(tmp_path)
    mf.add_poll(
        store,
        poll(
            T0,
            (21, "2026-10-25 02:25", "2026-10-25", 2, 2),
            (21, "2026-10-25 02:25", "2026-10-25", 12, 14),
        ),
    )
    assert read_day(store)["total_of_directions"].tolist() == [4, 26]


def test_a_poll_across_midnight_writes_both_local_days(tmp_path):
    store = mf.LocalStore(tmp_path)
    added = mf.add_poll(
        store,
        poll(
            T0,
            (61, "2026-10-25 12:59", "2026-10-25", 1, 0),  # 23:59 in Melbourne
            (61, "2026-10-25 13:00", "2026-10-26", 1, 0),  # 00:00 the next day
        ),
    )
    assert sorted(added) == [DAY, "minute_feed_2026-10-26.parquet"]


class RacingStore(mf.LocalStore):
    """Another machine writes the day file between our first read and our write."""

    def __init__(self, folder, other_rows):
        super().__init__(folder)
        self.other_rows, self.raced = other_rows, False

    def write(self, name, data, token):
        if not self.raced:
            self.raced = True
            mf.LocalStore.write(self, name, mf.to_bytes(self.other_rows), token)
        super().write(name, data, token)


def test_a_concurrent_write_is_merged_not_overwritten(tmp_path):
    theirs = poll(T0, (61, "2026-10-25 05:00", "2026-10-25", 1, 1))
    ours = poll(T1, (61, "2026-10-25 05:01", "2026-10-25", 2, 0))
    store = RacingStore(tmp_path, theirs)
    assert mf.merge_into(store, DAY, ours) == 1
    out = read_day(store)
    assert len(out) == 2  # their row survived our write


class FakeResponse:
    def __init__(self, status, content=b"", headers=None, body=None):
        self.status_code, self.content = status, content
        self.headers, self._body = headers or {}, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._body


class FakeBucket:
    """Just enough of the GCS JSON API: media get/upload with generations, and list."""

    def __init__(self):
        self.objects = {}  # name -> (bytes, generation)

    def get(self, url, params=None, timeout=None):
        if url.endswith("/o"):
            prefix = params["prefix"]
            items = [
                {"name": n, "size": str(len(b)), "generation": str(g)}
                for n, (b, g) in self.objects.items()
                if n.startswith(prefix)
            ]
            return FakeResponse(200, body={"items": items})
        name = url.rsplit("/o/", 1)[1].replace("%2F", "/")
        if name not in self.objects:
            return FakeResponse(404)
        data, gen = self.objects[name]
        return FakeResponse(200, data, {"x-goog-generation": str(gen)})

    def post(self, url, params=None, data=None, headers=None, timeout=None):
        name, want = params["name"], int(params["ifGenerationMatch"])
        have = self.objects.get(name, (None, 0))[1]
        if want != have:
            return FakeResponse(412)
        self.objects[name] = (data, have + 1)
        return FakeResponse(200)


def test_bucket_store_uses_generations_so_stale_writes_are_refused():
    bucket = FakeBucket()
    store = mf.GcsStore("gs://b/minute_feed", bucket)
    data, gen = store.read(DAY)
    assert (data, gen) == (None, 0)
    store.write(DAY, b"one", gen)
    assert bucket.objects["minute_feed/" + DAY] == (b"one", 1)
    with pytest.raises(mf.Conflict):
        store.write(DAY, b"stale", 0)  # written against the old generation


def test_polls_merge_into_the_bucket_and_sync_copies_them(tmp_path):
    bucket = FakeBucket()
    store = mf.GcsStore("gs://b/minute_feed", bucket)
    mf.add_poll(store, poll(T0, (61, "2026-10-25 05:00", "2026-10-25", 1, 1)))
    mf.add_poll(store, poll(T1, (61, "2026-10-25 05:01", "2026-10-25", 2, 0)))
    cfg = {"minute_feed": {"dataset": "d", "license": "CC BY 4.0"}}
    assert mf.sync(store, tmp_path, cfg) == 1
    assert len(pd.read_parquet(tmp_path / DAY)) == 2
    assert (tmp_path / "MANIFEST.json").exists()
    assert mf.sync(store, tmp_path, cfg) == 0  # already held


def test_no_credentials_skips_instead_of_failing(monkeypatch):
    from src.pipeline.dvc_push import MissingCredentials

    def no_creds():
        raise MissingCredentials("no credentials")

    monkeypatch.setattr(mf, "gcs_session", no_creds)
    monkeypatch.setattr("sys.argv", ["minute_feed", "--bucket"])
    with pytest.raises(SystemExit) as stop:
        mf.main()
    assert stop.value.code == mf.SKIP_EXIT_CODE


def test_each_hour_starts_with_a_full_poll_and_the_rest_fetch_recent_hours():
    assert mf.poll_since(datetime(2026, 10, 25, 7, 0, tzinfo=UTC), 3) is None
    assert mf.poll_since(datetime(2026, 10, 25, 7, 14, tzinfo=UTC), 3) is None
    since = mf.poll_since(datetime(2026, 10, 25, 7, 30, 42, tzinfo=UTC), 3)
    assert since == datetime(2026, 10, 25, 4, 30, tzinfo=UTC)


def test_a_recent_poll_asks_the_portal_only_for_rows_since(monkeypatch, tmp_path):
    asked = {}

    def fake_download(url, dest, params=None, **kw):
        asked["params"] = params
        export((61, "2026-10-25 05:00", "2026-10-25", 1, 1)).to_parquet(dest)
        return dest

    monkeypatch.setattr(mf, "download", fake_download)
    cfg = {
        "minute_feed": {"dataset": "d"},
        "melbourne_portal": {"base_url": "https://example.org"},
    }
    mf.fetch_poll(cfg, datetime(2026, 10, 25, 4, 30, tzinfo=UTC))
    assert asked["params"] == {
        "where": "sensing_datetime >= date'2026-10-25T04:30:00+00:00'"
    }
    mf.fetch_poll(cfg)
    assert asked["params"] is None


def test_an_empty_poll_writes_nothing(tmp_path):
    empty = poll(T0, (61, "2026-10-25 05:00", "2026-10-25", 1, 1)).iloc[:0]
    assert mf.add_poll(mf.LocalStore(tmp_path), empty) == {}
