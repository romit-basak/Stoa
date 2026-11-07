import numpy as np
import pandas as pd
import pytest

from src.features.sun import solar_position

MELB = (-37.8136, 144.9631)
BOSTON = (42.3398, -71.0892)


def at(*stamps):
    return pd.Series(pd.to_datetime(list(stamps), utc=True))


def noon_index(lat, lon, day):
    """The UTC minute of the day when the sun is highest, found by brute force."""
    minutes = pd.date_range(f"{day} 00:00", f"{day} 23:59", freq="min", tz="UTC")
    elev, az = solar_position(pd.Series(minutes), lat, lon)
    return elev.argmax(), elev, az


@pytest.mark.parametrize("lat,lon,expected", [(*BOSTON, 180.0), (*MELB, 0.0)])
def test_the_noon_sun_is_south_in_boston_and_north_in_melbourne(lat, lon, expected):
    i, _, az = noon_index(lat, lon, "2026-03-20")
    gap = min(abs(az[i] - expected), 360 - abs(az[i] - expected))
    assert gap < 2


def test_morning_sun_is_east_and_afternoon_sun_west():
    i, _, az = noon_index(*BOSTON, "2026-06-21")
    assert 45 < az[i - 180] < 135  # three hours before noon
    assert 225 < az[i + 180] < 315  # three hours after


def test_noon_elevation_matches_latitude_and_season():
    # At the equinox the noon sun stands at 90 minus the latitude.
    i, elev, _ = noon_index(*BOSTON, "2026-03-20")
    assert elev[i] == pytest.approx(90 - BOSTON[0], abs=1.0)
    # At the June solstice it is 23.4 degrees higher in Boston and lower in Melbourne.
    j, elev_b, _ = noon_index(*BOSTON, "2026-06-21")
    k, elev_m, _ = noon_index(*MELB, "2026-06-21")
    assert elev_b[j] == pytest.approx(90 - BOSTON[0] + 23.44, abs=1.0)
    assert elev_m[k] == pytest.approx(90 + MELB[0] - 23.44, abs=1.0)


def test_azimuth_stays_in_range():
    _, az = solar_position(at("2026-01-01 00:00", "2026-07-01 12:00"), *MELB)
    assert np.all((az >= 0) & (az < 360))
