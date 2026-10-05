from src.ingest import nyc


def test_fetch_paged_csv_concatenates_pages_with_one_header(tmp_path, monkeypatch):
    rows = [f"{i},x\n" for i in range(5)]

    def fake_download(url, dest, params=None, **kwargs):
        start = params["$offset"]
        dest.write_text("id,v\n" + "".join(rows[start : start + params["$limit"]]))
        return dest

    monkeypatch.setattr(nyc, "download", fake_download)
    dest = tmp_path / "out.csv"
    n = nyc.fetch_paged_csv("https://example.org/x.csv", dest, {}, page_size=2)

    assert n == 5
    assert dest.read_text() == "id,v\n" + "".join(rows)
    assert not list(tmp_path.glob("*.page")) and not list(tmp_path.glob("*.part"))


def test_nyc_config_is_evaluation_only_and_well_formed():
    cfg = nyc.load_config()["nyc_open_data"]
    assert cfg["datasets"]["bi_annual_pedestrian_counts"]["id"] == "cqsj-cfgu"
    assert all(d["format"] in {"csv", "geojson"} for d in cfg["datasets"].values())
    assert "Evaluation only" in nyc.LICENSE
