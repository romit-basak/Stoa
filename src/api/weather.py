"""Hourly weather for serving: Open-Meteo's forecast, in the shape training's archive has.

Training joins Open-Meteo's ERA5 archive (src/ingest/weather.py, read by
counts_hourly.read_weather). The archive lags real time by days, so predictions use the
forecast endpoint: same provider, same nine variables, but a forecast rather than a
reanalysis. That difference is the one known gap between served and trained inputs
(docs/api/serving.md). The forecast covers the past `past_days` and the next
`forecast_days` (Open-Meteo allows 92 and 16); hours outside that are refused.

Asked in GMT, so its timestamps are UTC with no daylight-saving offset to undo. A fetched
table is kept for `ttl_s` seconds, since a forecast changes only every hour or so.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import requests

from src.api.features import BadHours

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """An Open-Meteo response as a table indexed by UTC hour (as read_weather builds it)."""
    df = pd.DataFrame(payload["hourly"])
    df["ts_utc"] = (
        pd.to_datetime(df.pop("time"))
        - pd.Timedelta(seconds=payload.get("utc_offset_seconds", 0))
    ).dt.tz_localize("UTC")
    return df.drop_duplicates("ts_utc", keep="last").set_index("ts_utc").sort_index()


class ForecastWeather:
    def __init__(
        self,
        data_cfg: dict[str, Any],
        site: str,
        past_days: int = 92,
        forecast_days: int = 16,
        ttl_s: float = 3600,
        fetch: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        s = data_cfg["sites"][site]
        self.params = {
            "latitude": s["center"][1],
            "longitude": s["center"][0],
            "hourly": ",".join(data_cfg["weather"]["hourly"]),
            "timezone": "GMT",
            "past_days": past_days,
            "forecast_days": forecast_days,
        }
        self.ttl_s = ttl_s
        self.fetch = fetch or self._fetch
        self._table: pd.DataFrame | None = None
        self._at = 0.0

    @staticmethod
    def _fetch(params: dict[str, Any]) -> dict[str, Any]:
        r = requests.get(FORECAST_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def table(self) -> pd.DataFrame:
        if self._table is None or time.monotonic() - self._at > self.ttl_s:
            table = to_frame(self.fetch(self.params))
            table.attrs["source"] = "Open-Meteo forecast (CC BY 4.0)"
            self._table, self._at = table, time.monotonic()
        return self._table

    def for_hours(self, ts_utc: pd.Series) -> pd.DataFrame:
        """The table, after checking it covers every hour asked for (and the hour after it,
        where the interval variables are stamped)."""
        table = self.table()
        first, last = table.index.min(), table.index.max() - pd.Timedelta(hours=1)
        outside = ~ts_utc.between(first, last)
        if outside.any():
            raise BadHours(
                f"{int(outside.sum())} hour(s) outside the weather forecast's range, "
                f"{first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC"
            )
        return table
