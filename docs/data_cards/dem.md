# Terrain (DEM / DSM): Melbourne

| | |
|---|---|
| **Role** | Slope where CoM footpath-steepness is missing; ground surface for viewshed ray-casting (with CoM building footprints on top) |
| **Access** | `python -m src.ingest.dem --site melbourne` → `data/raw/dem/` |

## What we pulled

| File | Product | Res | Type | License | Notes |
|---|---|---|---|---|---|
| `melbourne_ga_lidar_5m.tif` | Geoscience Australia "DEM of Australia derived from LiDAR 5 m" (`DEM_LiDAR_5m_2025`), WCS clip | 5 m (0.00005°), EPSG:4283 | **Bare earth (DTM)** | CC BY 4.0 (from catalog page; licence doc not read) | **Primary.** 1720×2100 px, 15 MB. Built from 236 surveys 2001–2015; which survey covers the CBD is unverified (likely the 2007–09 Port Phillip LiDAR). |
| `Copernicus_DSM_COG_10_S38_00_E144_00_DEM.tif` | Copernicus GLO-30 | 30 m | **Surface (DSM)** | Copernicus DEM licence (free, attribution) | Comparison only |
| `S38E144.hgt.gz` | SRTM 1″ (AWS Terrain Tiles) | 30 m | Surface (C-band, ~2000) | Public domain | Comparison only |

## Validation spot checks (m, AHD)

| Point | GA 5 m | GLO-30 | SRTM |
|---|---|---|---|
| CBD (Swanston/Collins) | 9.9 | 33.2 | 37.0 |
| Flagstaff Gardens | 31.0 | 36.4 | 42.0 |
| Docklands | 2.8 | 4.8 | 6.0 |
| Princes Park (Carlton N.) | **0.0 (void)** | 46.1 | 48.0 |

The 30 m products read 20–27 m high in the CBD because they measure rooftops. Do not use them for slope or ground height in the city.

## GA 5 m known issues

- **15.7% of pixels are exactly 0.** 70% of those are water (WorldCover class 80: Yarra, Maribyrnong, docks), but **4.7% of the study area is a void on land** (76% of it built-up), notably Princes Park / Carlton North at the northern edge and patches in the south-west (Port Melbourne / Fishermans Bend). The file declares no nodata value. Set `0 → nodata` in features and fill voids from (a) CoM `footpath-steepness` / building `structure_min_elevation` ground points or (b) Vicmap Elevation Metro contours (CC BY 4.0, ArcGIS FeatureServer). Use the 30 m DSMs only in parks with no buildings.
- 202 isolated pits below -5 m (min -80.6). Clip or median-filter them.
- Only 1 of 100 outdoor count sensors (Sandridge Bridge, over the river) sits on a void pixel.
- Pre-2018 ground. Recent large earthworks (e.g. Metro Tunnel station precincts) are not reflected.

## Options considered and rejected

| Product | Why not |
|---|---|
| **FABDEM** (30 m, buildings/forest removed) | **CC BY-NC-SA 4.0.** Non-commercial; violates project rule 5 for the product pipeline. |
| Vicmap Elevation 1 m DEM | Restricted (DALA / paid reseller) |
| Greater Melbourne LiDAR 2017–18 (1 m DTM) | Licensed via resellers; ELVIS copies are CC BY-NC 4.0. Validation-only at most, and only after a team decision. |
| Vicmap Buildings | Restricted (DALA) |
| Google Open Buildings 2.5D | No Australia coverage |

## Higher resolution, if needed

- **Sub-metre ground:** classify ground points in the CoM 2018 point cloud (CC BY, 7.5 cm) with PDAL SMRF and rasterise to 0.5–1 m. Multi-GB download; only worth it if the viewshed engine needs kerb-level detail.
- **DSM for viewshed:** CoM 2018 DSM 0.1 m (12 GB, CC BY) gives roofs and trees directly, but extruded CoM footprints plus tree points are lighter and newer (2023).

## Boston / NEU

Pulled 2026-10-04: a 1 m USGS 3DEP clip from the ImageServer (`data/raw/dem/neu_boston_usgs_3dep_1m.tif`), avoiding the ~410 MB staged tiles. Details in `neu_boston.md`.
