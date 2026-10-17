from datetime import date

import pandas as pd

from src.ingest import melbourne_portal

KEY = ["location_id", "sensing_date", "hourday"]


def test_merge_rolling_keeps_old_rows_and_prefers_fresh(tmp_path):
    held = tmp_path / "counts.parquet"
    fresh = tmp_path / "counts.fresh.parquet"
    pd.DataFrame({"id": [1, 2, 3], "n": [10, 20, 30]}).to_parquet(held)
    # The portal window has moved on: row 1 dropped, row 3 corrected, row 4 new.
    pd.DataFrame({"id": [2, 3, 4], "n": [20, 31, 40]}).to_parquet(fresh)

    assert melbourne_portal.merge_rolling(held, fresh, "id") == 4
    out = pd.read_parquet(held)
    assert out["id"].tolist() == [1, 2, 3, 4]
    assert out["n"].tolist() == [10, 20, 31, 40]
    assert not fresh.exists()


def test_merge_rolling_keeps_rows_whose_portal_ids_collide(tmp_path):
    # The portal's id is location + hour + date glued together, so location 4 at hour 20
    # and location 42 at hour 2 on the same day share id 42020261015. Both rows are real.
    day = date(2026, 10, 15)
    rows = pd.DataFrame(
        {
            "id": [42020261015, 42020261015],
            "location_id": [4, 42],
            "sensing_date": [day, day],
            "hourday": [20, 2],
            "pedestriancount": [1968, 5],
        }
    )
    held = tmp_path / "counts.parquet"
    fresh = tmp_path / "counts.fresh.parquet"
    rows.iloc[:0].to_parquet(held)
    rows.to_parquet(fresh)

    assert melbourne_portal.merge_rolling(held, fresh, KEY) == 2
    out = pd.read_parquet(held).set_index("location_id")["pedestriancount"]
    assert out.to_dict() == {4: 1968, 42: 5}


def test_merge_rolling_on_composite_key_updates_the_partial_day(tmp_path):
    held = tmp_path / "counts.parquet"
    fresh = tmp_path / "counts.fresh.parquet"
    old_day, today = date(2026, 10, 1), date(2026, 10, 16)
    pd.DataFrame(
        {
            "location_id": [1, 1],
            "sensing_date": [old_day, today],
            "hourday": [8, 8],
            "pedestriancount": [100, 7],  # today's hour 8 was still filling in
        }
    ).to_parquet(held)
    pd.DataFrame(
        {
            "location_id": [1, 1],
            "sensing_date": [today, today],
            "hourday": [8, 9],
            "pedestriancount": [250, 300],
        }
    ).to_parquet(fresh)

    assert melbourne_portal.merge_rolling(held, fresh, KEY) == 3
    out = pd.read_parquet(held)
    assert out["pedestriancount"].tolist() == [100, 250, 300]


def test_recent_rows_filter_starts_overlap_days_before_latest_held(tmp_path):
    held = tmp_path / "counts.parquet"
    pd.DataFrame(
        {"sensing_date": [date(2026, 9, 30), date(2026, 10, 16), date(2026, 10, 2)]}
    ).to_parquet(held)

    where = melbourne_portal.recent_rows_filter(held, "sensing_date", 2)
    assert where == "sensing_date >= date'2026-10-14'"


def test_counts_table_is_configured_as_rolling_on_a_composite_key():
    cfg = melbourne_portal.load_config()["melbourne_portal"]
    rolling = cfg["rolling_window"][
        "pedestrian-counting-system-monthly-counts-per-hour"
    ]
    assert rolling["key"] == KEY
    assert rolling["date_field"] == "sensing_date"
    assert rolling["overlap_days"] >= 1
