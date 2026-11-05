from dataclasses import replace

import pytest

from src.features import osm as cli
from src.features.common import site_settings


@pytest.fixture
def site(tmp_path, osm_file):
    s = site_settings("melbourne")
    return replace(
        s, interim=tmp_path / "interim", processed=tmp_path / "processed", pbf=osm_file
    )


def touch_outputs(site, stage):
    for p in cli.stage_outputs(site)[stage]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_stage_is_current_only_with_matching_stamp_and_outputs(site, fcfg):
    from src.ingest.common import load_config

    fp = cli.fingerprints(site, fcfg, load_config())
    assert not cli.is_current(site, "graph", fp["graph"])  # never ran
    cli.write_stamp(site, "graph", fp["graph"])
    assert not cli.is_current(site, "graph", fp["graph"])  # outputs missing
    touch_outputs(site, "graph")
    assert cli.is_current(site, "graph", fp["graph"])
    assert not cli.is_current(site, "graph", "something else")


def test_fingerprints_change_with_inputs(site, fcfg):
    from src.ingest.common import load_config

    data_cfg = load_config()
    base = cli.fingerprints(site, fcfg, data_cfg)
    # A config change invalidates every stage, since each builds on the one before.
    changed = {**fcfg, "amenities": {**fcfg["amenities"], "snap_max_m": 49}}
    after = cli.fingerprints(site, changed, data_cfg)
    assert all(base[s] != after[s] for s in cli.STAGES)
    # A different extract does too.
    other = replace(site, pbf=site.pbf.with_name("missing.osm.pbf"))
    assert cli.fingerprints(other, fcfg, data_cfg)["extract"] != base["extract"]
    # The same inputs give the same fingerprints.
    assert cli.fingerprints(site, fcfg, data_cfg) == base


def test_downstream_settings_leave_the_osm_stages_current(site, fcfg):
    from src.ingest.common import load_config

    data_cfg = load_config()
    base = cli.fingerprints(site, fcfg, data_cfg)
    changed = {**fcfg, "training_frame": {"anything": 1}, "minute_profiles": {}}
    assert cli.fingerprints(site, changed, data_cfg) == base


def test_no_osm_stage_imports_a_downstream_module():
    import ast

    names = {n.removesuffix(".py") for n in cli.DOWNSTREAM_CODE}
    for f in sorted(cli.FEATURE_CODE.glob("*.py")):
        if f.name in cli.DOWNSTREAM_CODE:
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.rsplit(".", 1)[-1] not in names, (
                    f.name,
                    node.module,
                )
                if node.module == "src.features":
                    assert not {a.name for a in node.names} & names, f.name
