from src.ingest.osm import should_download


def test_extract_is_pinned_unless_refreshed(tmp_path):
    dest = tmp_path / "melbourne.osm.pbf"
    assert should_download(dest, refresh=False)  # nothing yet
    dest.write_bytes(b"pbf")
    assert not should_download(dest, refresh=False)
    assert should_download(dest, refresh=True)
