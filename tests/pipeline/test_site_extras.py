import sys

from src.ingest.common import load_config
from src.pipeline import site_extras


def test_extras_run_each_listed_ingest_for_the_site():
    cfg = {"sites": {"a": {"extra_ingest": ["boston_buildings", "imagery"]}, "b": {}}}
    assert site_extras.commands("a", cfg) == [
        [sys.executable, "-m", "src.ingest.boston_buildings", "--site", "a"],
        [sys.executable, "-m", "src.ingest.imagery", "--site", "a"],
    ]
    assert site_extras.commands("b", cfg) == []


def test_every_listed_extra_is_an_ingest_module_with_a_site_option():
    cfg = load_config()
    from src.ingest.common import REPO_ROOT

    for site, s in cfg["sites"].items():
        for name in s.get("extra_ingest", []):
            module = REPO_ROOT / "src" / "ingest" / f"{name}.py"
            assert module.exists(), (site, name)
            assert '"--site"' in module.read_text(), (site, name)
