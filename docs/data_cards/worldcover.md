# ESA WorldCover 10 m (2021, v200)

| | |
|---|---|
| **Role** | Land-cover context per segment (tree cover, built-up, water, grass), mainly for the NEU campus and areas outside CoM layers |
| **Provider** | European Space Agency, WorldCover consortium |
| **License** | CC BY 4.0. Attribution: "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium" |
| **Access** | `python -m src.ingest.worldcover --site melbourne` (public AWS bucket, no auth) |
| **Files** | `ESA_WorldCover_10m_2021_v200_S39E144_Map.tif` (full 3°×3° tile, 66 MB); `melbourne_worldcover_2021.tif` (clipped to study bbox, 1031×1260 px) |
| **CRS / resolution** | EPSG:4326, 1/12000° (~10 m) |

## Class shares in the Melbourne study bbox

| Class | Code | Share |
|---|---|---|
| Built-up | 50 | 62.9% |
| Tree cover | 10 | 14.5% |
| Permanent water | 80 | 11.9% |
| Grassland | 30 | 7.9% |
| Bare / sparse | 60 | 2.2% |
| Herbaceous wetland | 90 | 0.6% |
| Cropland | 40 | 0.1% |

## Other sites

- `neu_boston` (tile N42W072): built-up 78.4%, tree cover 18.3%, grass 2.2%.
- `nyc` (tile N39W075, count-location extent): built-up 52.1%, water 24.1%, tree cover 15.5%, grass 5.0%.

## Known issues

- At 10 m the data is too coarse for street-level shade. Street trees in the CBD mostly show as built-up. For Melbourne, prefer CoM tree canopies 2021 (polygons) and trees-with-dimensions. WorldCover is the portable fallback for sites without such data (e.g. Boston, together with MassGIS).
- Only one year (2021) exists for v200, so the data is static.
- Dynamic World (time series, also CC BY 4.0) and Sentinel-2 need Google Earth Engine or an STAC pipeline. Not pulled yet.
