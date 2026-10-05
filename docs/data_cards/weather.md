# Hourly weather (Melbourne): Open-Meteo historical archive

| | |
|---|---|
| **Role** | Hourly weather features (temperature, apparent temperature, rain, wind, cloud, radiation, day/night) |
| **Provider** | Open-Meteo archive API, built on ECMWF ERA5 / ERA5-Land reanalysis |
| **License** | Data CC BY 4.0 (attribute Open-Meteo and Copernicus/ECMWF). **The free API tier is for non-commercial use only.** That's fine for this course project; a commercial product needs an Open-Meteo subscription or a pull straight from Copernicus CDS (ERA5 licence allows commercial use). |
| **Access** | `python -m src.ingest.weather --site melbourne`, one JSON per year: `data/raw/weather/melbourne_openmeteo_YYYY.json` |
| **Coverage** | 2009-01-01 00:00 → 2026-09-26 23:00 local time, 155,472 hourly rows, 0 nulls, 0 duplicate timestamps |
| **Location** | Request at CBD centre (-37.8136, 144.9631); API snapped to grid cell (-37.786, 144.940), elevation 19 m |
| **Time zone** | `Australia/Melbourne` local time, so it joins directly on (sensing_date, hourday) |

## Other sites

Same variables, local time America/New_York, no nulls or duplicate timestamps:
- `neu_boston_openmeteo_YYYY.json`: 2009-01-01 → 2026-09-28 (155,520 h).
- `nyc_openmeteo_YYYY.json`: 2007-01-01 → 2026-09-28 (173,064 h), starting early to cover the NYC bi-annual counts. Point at Midtown; one point for all five boroughs is coarse but fine for period-level evaluation.

The API rate-limits bursts (HTTP 429); `common.download` retries with backoff.

## Variables

`temperature_2m` (°C), `apparent_temperature` (°C), `relative_humidity_2m` (%), `precipitation` / `rain` (mm), `cloud_cover` (%), `wind_speed_10m` (km/h), `shortwave_radiation` (W/m²), `is_day` (0/1)

## Known issues

- Reanalysis is a modelled grid value (~9–11 km), not a station observation. Rain timing can be off by an hour, and short CBD showers are smoothed. One value covers the whole study area, which is fine at this scale.
- The archive lags real time by ~5 days. Live inference needs a forecast endpoint instead.
- DST: local-time series have a repeated hour in April and a missing hour in October. Join on the same local-time convention as the counts and check those days.
- Station alternative: BoM Melbourne (Olympic Park) 086338. The CoM portal copy (`bom-weather-at-olympic-park`) is only a short 2019–2020 30-minute feed with no stated license. BoM long-term hourly data is not freely downloadable. The project's weather source is still listed as **TBD** in CLAUDE.md. This is the proposed default.
