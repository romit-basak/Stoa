import numpy as np
import pandas as pd
import pytest

from src.simulation import animate

TZ = animate.TZ


def test_walkers_move_along_their_route_and_vanish_off_its_ends():
    line = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    along = np.array([0.0, 100.0, 200.0])
    starts = np.array([0.0, 50.0, 200.0])
    xy = animate.positions(along, line, starts, now_s=100.0, speed=1.5)
    assert xy[0].tolist() == pytest.approx([100.0, 50.0])
    assert xy[1].tolist() == pytest.approx([75.0, 0.0])
    assert np.isnan(xy[2]).all()
    later = animate.positions(along, line, starts, now_s=1000.0, speed=1.5)
    assert np.isnan(later).all()


def test_each_sensor_shows_the_observed_count_of_the_current_block():
    block = pd.Timestamp("2026-10-27 23:00", tz=TZ)
    series = pd.DataFrame({"location_id": [161], "block": [block], "total": [452]})
    counts = animate.block_counts(series, 5)
    inside = block + pd.Timedelta(minutes=3)
    assert animate.observed_now(counts, 161, inside, 5) == 452
    assert animate.observed_now(counts, 161, block + pd.Timedelta(minutes=5), 5) == 0
    assert animate.observed_now(counts, 136, inside, 5) == 0


def test_the_gif_is_written(tmp_path):
    line = np.array([[0.0, 0.0], [300.0, 0.0]])
    routes = {0: (line, np.array([0.0, 300.0]))}
    walkers = pd.DataFrame({"route": [0, 0, 0], "start_s": [0.0, 30.0, 60.0]})
    first = pd.Timestamp("2026-10-27 22:30", tz=TZ)
    frames = list(pd.date_range(first, periods=4, freq="1min"))
    sensors = [{"id": 161, "x": 150.0, "y": 0.0}]
    counts = {(161, first): 12}
    cfg = {
        "width_px": 400,
        "dpi": 50,
        "margin_m": 20,
        "sample_every": 5,
        "speed_mps": 1.3,
        "block_minutes": 5,
        "fps": 4,
    }
    path = tmp_path / "a.gif"
    animate.render(
        [line], routes, walkers, sensors, counts, frames, cfg, "caption", path
    )
    assert path.exists() and path.stat().st_size > 1000
