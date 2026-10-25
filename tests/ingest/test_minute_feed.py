from datetime import UTC, datetime

import pandas as pd
import pytest

from src.ingest import minute_feed

T0 = datetime(2026, 10, 25, 5, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 25, 5, 15, tzinfo=UTC)


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


def test_normalise_fixes_types_and_adds_the_poll_time():
    out = minute_feed.normalise(
        export((61, "2026-10-25 05:04", "2026-10-25", 2, 4)), T0
    )
    assert out.columns.tolist() == minute_feed.COLUMNS + ["polled_at"]
    assert str(out["location_id"].dtype) == "int32"
    assert out["sensing_datetime"].dt.tz is not None
    assert out.loc[0, "sensing_time"] == "16:04"  # local clock, as published
    assert out.loc[0, "polled_at"] == pd.Timestamp(T0)


def test_normalise_rejects_a_changed_export():
    with pytest.raises(ValueError, match="total_of_directions"):
        minute_feed.normalise(
            export((61, "2026-10-25 05:04", "2026-10-25", 2, 4)).drop(
                columns="total_of_directions"
            ),
            T0,
        )


def test_overlapping_polls_keep_one_row_with_the_first_poll_time(tmp_path):
    a = minute_feed.normalise(
        export(
            (61, "2026-10-25 05:00", "2026-10-25", 1, 1),
            (61, "2026-10-25 05:01", "2026-10-25", 2, 0),
        ),
        T0,
    )
    b = minute_feed.normalise(
        export(
            (61, "2026-10-25 05:01", "2026-10-25", 2, 0),  # seen before
            (61, "2026-10-25 05:02", "2026-10-25", 0, 3),  # new
        ),
        T1,
    )
    assert minute_feed.add_poll(tmp_path, a)
    assert minute_feed.add_poll(tmp_path, b)
    out = pd.read_parquet(minute_feed.day_file(tmp_path, "2026-10-25"))
    assert len(out) == 3
    seen_twice = out[
        out["sensing_datetime"] == pd.Timestamp("2026-10-25 05:01", tz="UTC")
    ]
    assert seen_twice["polled_at"].tolist() == [pd.Timestamp(T0)]
    # A repeat of a poll changes nothing, so nothing is uploaded again.
    assert minute_feed.add_poll(tmp_path, b) == []


def test_two_different_rows_for_one_minute_are_both_kept(tmp_path):
    # The portal sometimes publishes two readings for a sensor and minute; keep what it said.
    poll = minute_feed.normalise(
        export(
            (21, "2026-10-25 02:25", "2026-10-25", 2, 2),
            (21, "2026-10-25 02:25", "2026-10-25", 12, 14),
        ),
        T0,
    )
    minute_feed.add_poll(tmp_path, poll)
    out = pd.read_parquet(minute_feed.day_file(tmp_path, "2026-10-25"))
    assert out["total_of_directions"].tolist() == [4, 26]


def test_a_poll_across_midnight_writes_both_local_days(tmp_path):
    poll = minute_feed.normalise(
        export(
            (61, "2026-10-25 12:59", "2026-10-25", 1, 0),  # 23:59 in Melbourne
            (61, "2026-10-25 13:00", "2026-10-26", 1, 0),  # 00:00 the next day
        ),
        T0,
    )
    changed = minute_feed.add_poll(tmp_path, poll)
    assert [p.name for p in changed] == [
        "minute_feed_2026-10-25.parquet",
        "minute_feed_2026-10-26.parquet",
    ]


def test_sync_downloads_only_new_or_changed_day_files(tmp_path, monkeypatch):
    held = tmp_path / "minute_feed_2026-10-25.parquet"
    held.write_bytes(b"x" * 10)
    assets = [
        {
            "name": held.name,
            "size": 10,
            "browser_download_url": "u1",
            "updated_at": "t",
        },
        {
            "name": "minute_feed_2026-10-26.parquet",
            "size": 5,
            "browser_download_url": "u2",
            "updated_at": "t",
        },
    ]
    monkeypatch.setattr(minute_feed, "release_assets", lambda repo, prefix: assets)
    fetched = []

    def fake_download(url, dest, **kw):
        fetched.append(url)
        dest.write_bytes(b"y" * 5)
        return dest

    monkeypatch.setattr(minute_feed, "download", fake_download)
    cfg = {
        "minute_feed": {
            "dataset": "d",
            "license": "CC BY 4.0",
            "github_repo": "o/r",
            "release_prefix": "minute-feed-",
        }
    }
    assert minute_feed.sync(cfg, tmp_path) == 1
    assert fetched == ["u2"]
    assert (tmp_path / "MANIFEST.json").exists()


def test_release_assets_keep_only_day_files_of_archive_releases(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "tag_name": "minute-feed-2026",
                    "assets": [
                        {"name": "minute_feed_2026-10-25.parquet"},
                        {"name": "MANIFEST.json"},
                    ],
                },
                {"tag_name": "v1.0", "assets": [{"name": "minute_feed_x.parquet"}]},
            ]

    monkeypatch.setattr(minute_feed.requests, "get", lambda *a, **k: Response())
    names = [a["name"] for a in minute_feed.release_assets("o/r", "minute-feed-")]
    assert names == ["minute_feed_2026-10-25.parquet"]
