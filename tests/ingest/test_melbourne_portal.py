import pandas as pd

from src.ingest import melbourne_portal


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


def test_counts_table_is_configured_as_rolling():
    cfg = melbourne_portal.load_config()["melbourne_portal"]
    assert (
        cfg["rolling_window"]["pedestrian-counting-system-monthly-counts-per-hour"]
        == "id"
    )
