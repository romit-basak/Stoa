# Data licenses

Check this file before adding any dataset (CLAUDE.md rule 5). **Train** = may feed product model training/features. **Validate** = evaluation only. **Excluded** = do not download into the pipeline.

| Dataset | License | Status | Attribution / obligations |
|---|---|---|---|
| CoM pedestrian counts (hourly + 2009–2022 archive) | **Not stated in portal metadata** (likely CC BY 4.0) | Train, **pending confirmation** | Ask the CoM Open Data Team in writing; record the answer here |
| CoM sensor locations, pedestrian network, footpaths, footpath steepness, buildings, CLUE, trees/canopy, POIs, transit, events, microclimate, DSM, point cloud | CC BY 4.0 | Train | "Source: City of Melbourne Open Data" |
| CoM `bom-weather-at-olympic-park` | Not stated (BoM data) | Validate | BoM copyright; check BoM terms before any reuse |
| OpenStreetMap | ODbL 1.0 | Train | "© OpenStreetMap contributors". **Share-alike** if we distribute a derived database (e.g. a published graph). Predictions are produced works: attribution only. |
| ESA WorldCover 2021 | CC BY 4.0 | Train | "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021)" |
| GA 5 m LiDAR DEM | CC BY 4.0 (catalog page) | Train | "© Commonwealth of Australia (Geoscience Australia)". Confirm against the licence document. |
| Copernicus GLO-30 | Copernicus DEM licence (free incl. commercial) | Train (fallback) | "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus 2014–2018, provided under COPERNICUS by the EU and ESA" |
| SRTM 1″ | Public domain | Train (fallback) | Credit NASA/USGS |
| Open-Meteo archive (ERA5) | Data CC BY 4.0; **free API tier non-commercial** | Train (course project) | Attribute Open-Meteo + Copernicus/ECMWF. A commercial product needs a subscription or a direct ERA5 pull from CDS. |
| Squiggle AFL games | No published licence (public fixture facts) | Train, low risk | Credit Squiggle; don't redistribute bulk dumps |
| USGS 3DEP (Boston) | Public domain | Train | Credit USGS |
| City of Boston buildings with roof breaks | PDDL | Train | — |
| NAIP | Public domain | Train | — |
| MassGIS imagery / structures | Public record (credit requested) | **Verify** | "MassGIS (Bureau of Geographic Information)" |
| Overture buildings | ODbL (mixed) | Train (fill-in only) | Same share-alike caveat as OSM |
| **FABDEM** | **CC BY-NC-SA 4.0** | **Excluded** | Non-commercial |
| **Vicmap Elevation 1 m / Greater Melbourne LiDAR 2017–18** | Restricted / CC BY-NC (ELVIS) | **Excluded** (validation only after a team decision) | — |
| **Vicmap Buildings** | Restricted (DALA) | **Excluded** | — |
| **Stanford Drone Dataset** | Non-commercial | **Validate only, never train** | — |
