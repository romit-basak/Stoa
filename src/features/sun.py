"""Where the sun is: elevation and azimuth for UTC times at a place.

NOAA's simplified solar position equations (fractional year, equation of time, declination),
accurate to well under a degree, which is plenty for hourly shade and daylight features.
Azimuth is measured clockwise from north, so the noon sun is at 180 in the northern hemisphere
and near 0 in the southern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _declination_and_hour_angle(
    ts_utc: pd.Series, lon: float
) -> tuple[np.ndarray, np.ndarray]:
    t = pd.DatetimeIndex(ts_utc)
    doy = t.day_of_year.to_numpy()
    hours = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    gamma = 2 * np.pi / 365 * (doy - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    true_solar_min = hours * 60 + eqtime + 4 * lon
    hour_angle = np.radians(true_solar_min / 4 - 180)
    return decl, hour_angle


def solar_position(
    ts_utc: pd.Series, lat: float, lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """(elevation, azimuth) in degrees; azimuth clockwise from north, 0-360."""
    decl, ha = _declination_and_hour_angle(ts_utc, lon)
    phi = np.radians(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    elevation = 90 - np.degrees(np.arccos(np.clip(cos_zen, -1, 1)))
    azimuth = np.degrees(
        np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(decl) * np.cos(phi))
    )
    return elevation, (azimuth + 180.0) % 360.0


def solar_elevation(ts_utc: pd.Series, lat: float, lon: float) -> np.ndarray:
    """Sun elevation in degrees."""
    return solar_position(ts_utc, lat, lon)[0]
