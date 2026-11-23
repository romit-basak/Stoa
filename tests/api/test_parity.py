"""Served features must equal trained features, column for column, on real training rows.

Training builds a row's features with training_frame (eligible rows, add_derived, then
design_matrix); the API builds them with src/api/features.py from only the segment and the
hour. This test samples training rows, builds them both ways and compares every column of
both variants. Weather comes from the same archive files on both sides, so a difference
means the feature code disagrees, not the weather source.

Needs the DVC data (`dvc pull`), so it skips in CI, which has none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import counts_hourly as ch
from src.features import training_frame as tf
from src.features.common import load_features_config
from src.ingest.common import REPO_ROOT, load_config

FEATS = REPO_ROOT / "data" / "processed" / "features" / "melbourne"
NEEDED = [
    "counts_hourly.parquet",
    "osm_segment_features.parquet",
    "segment_steepness.parquet",
    "sensor_osm_features.parquet",
]
pytestmark = pytest.mark.skipif(
    not all((FEATS / f).exists() for f in NEEDED),
    reason="needs the DVC data (dvc pull) and segment_steepness",
)
SAMPLE = 3000


@pytest.fixture(scope="module")
def tables():
    from src.api.features import SiteTables

    return SiteTables.load("melbourne")


@pytest.fixture(scope="module")
def sample():
    """Eligible training rows, as training_frame.load() selects them, sampled."""
    rich = tf.row_features("rich")
    keep = ["location_id", "ts_utc", "ts_local", "date_local", "pedestriancount"]
    keep += ["has_location", "location_type", "suspect_outage_zero"]
    keep += ["post_metro_tunnel", "sun_elevation_deg", "sun_azimuth_deg"]
    hourly = pd.read_parquet(
        FEATS / "counts_hourly.parquet", columns=list(dict.fromkeys(keep + rich))
    )
    rows = tf.eligible(hourly)
    rng = np.random.default_rng(7)
    pick = np.sort(rng.choice(rows, size=SAMPLE, replace=False))
    return hourly.iloc[pick].reset_index(drop=True)


@pytest.fixture(scope="module")
def weather():
    data_cfg = load_config()
    w = ch.read_weather(REPO_ROOT / data_cfg["raw_dir"] / "weather", "melbourne")
    w.attrs["source"] = "open-meteo archive (ERA5)"
    return w


def trained(sample: pd.DataFrame, variant: str) -> pd.DataFrame:
    """The training code path, on the sampled rows."""
    data_cfg, fcfg = load_config(), load_features_config()
    hourly = tf.add_derived(sample, "melbourne", data_cfg, fcfg)
    osm = tf.sensor_features(
        pd.read_parquet(FEATS / "sensor_osm_features.parquet"), variant
    )
    frame = tf.Frame(
        hourly,
        osm,
        tf.row_features(variant) + tf.derived_features(variant),
        fcfg["built_form"]["max_distance_m"],
    )
    return pd.DataFrame(frame.matrix(), columns=frame.columns)


def served(tables, sample: pd.DataFrame, weather, variant: str) -> pd.DataFrame:
    from src.api.features import features_for_rows

    seg_of = pd.read_parquet(
        FEATS / "sensor_osm_features.parquet", columns=["location_id", "segment_id"]
    ).set_index("location_id")["segment_id"]
    rows = pd.DataFrame(
        {
            "segment_id": sample["location_id"].map(seg_of).to_numpy(),
            "ts_local": sample["ts_local"].to_numpy(),
            "ts_utc": sample["ts_utc"].to_numpy(),
        }
    )
    rows["ts_local"] = sample["ts_local"].reset_index(drop=True)
    rows["ts_utc"] = sample["ts_utc"].reset_index(drop=True)
    return features_for_rows(tables, rows, weather, variant)


def own_steepness_rows(sample: pd.DataFrame) -> np.ndarray:
    """Rows whose segment takes this sensor's slope: where two sensors share a segment,
    segment_steepness keeps the lowest location_id's, so the other sensor differs there."""
    st = pd.read_parquet(
        FEATS / "segment_steepness.parquet",
        columns=["segment_id", "sensor_location_id"],
    )
    seg_of = pd.read_parquet(
        FEATS / "sensor_osm_features.parquet", columns=["location_id", "segment_id"]
    ).set_index("location_id")["segment_id"]
    owner = st.set_index("segment_id")["sensor_location_id"]
    seg = sample["location_id"].map(seg_of)
    return (
        (seg.map(owner).astype("Int64") == sample["location_id"])
        .fillna(False)
        .to_numpy(dtype=bool)
    )


@pytest.mark.parametrize("variant", ["rich", "portable"])
def test_served_features_equal_trained(tables, sample, weather, variant):
    want = trained(sample, variant)
    got = served(tables, sample, weather, variant)
    assert list(got.columns) == list(want.columns)
    own = own_steepness_rows(sample)
    assert own.mean() > 0.9  # nearly every sampled sensor owns its segment's slope
    bad = {}
    for c in want.columns:
        a, b = want[c].to_numpy(), got[c].to_numpy()
        rows = (
            own
            if c in ("slope_pct", "slope_pct_local_mean", "elevation_m")
            else slice(None)
        )
        same = np.isclose(a[rows], b[rows], rtol=1e-6, atol=1e-6, equal_nan=True)
        if not same.all():
            bad[c] = int((~same).sum())
    assert not bad, f"columns that differ (rows): {bad}"
