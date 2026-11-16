import numpy as np
import pandas as pd

from src.simulation import figures


def test_clock_turns_minutes_after_22_into_times():
    assert figures.clock(0) == "22:00"
    assert figures.clock(95) == "23:35"
    assert figures.clock(150) == "00:30"


def test_minutes_after_start_counts_from_22_local():
    blocks = pd.Series(
        pd.to_datetime(["2026-10-27 22:00", "2026-10-27 23:05"]).tz_localize(figures.TZ)
    )
    assert figures.minutes_after_start(blocks, "2026-10-27").tolist() == [0.0, 65.0]


def test_the_figures_are_written(tmp_path):
    line = np.array([[0.0, 0.0], [100.0, 50.0], [200.0, 50.0]])
    other = np.array([[0.0, 0.0], [0.0, 150.0]])
    written = figures.walkers_map(
        streets=[line, other],
        routes=[([line], 120.0), ([other], 5.0)],
        sensors=[{"x": 100.0, "y": 50.0, "label": "161", "detail": "161: 10 / 9"}],
        venue=(0.0, 0.0),
        destinations={"station": (200.0, 50.0)},
        title="test",
        path=tmp_path / "map.png",
    )
    minutes = np.arange(0, 60, 5)
    counts = np.linspace(1, 20, len(minutes))
    panel = {
        "night": "2026-10-27",
        "obs_minutes": minutes,
        "obs_counts": counts,
        "sim_minutes": minutes,
        "sim_counts": counts[::-1],
        "correlation": -1.0,
    }
    nights = figures.nights_figure([panel, panel], 161, tmp_path / "nights.png")
    seen = figures.observed_figure(
        {161: (minutes, counts), 136: (minutes, counts / 2)},
        "2026-10-27",
        tmp_path / "observed.png",
    )
    for path in (written, nights, seen):
        assert path.exists() and path.stat().st_size > 1000
