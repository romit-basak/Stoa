# Data licenses

Check this file before adding any dataset (CLAUDE.md rule 5). **Train** = may feed product model training/features. **Validate** = evaluation only. **Validate + demo** = evaluation, plus fine-tuning for a demonstration that is never shipped (nothing derived from it is distributed). **Excluded** = do not download into the pipeline. A mirror can't relicense data: record the original licence.

| Dataset | License | Status | Attribution / obligations |
|---|---|---|---|
| CoM pedestrian counts (hourly + 2009–2022 archive) | CC BY 4.0 under the portal-wide terms ("Our open data is provided under a creative commons licence … free to share and adapt our data for any purpose, provided you give the appropriate credit", About our data page; confirmed 2026-10-04). The dataset's own licence field is empty; every other CoM dataset lists CC BY 4.0 | Train | "Source: City of Melbourne Open Data" |
| CoM sensor locations, pedestrian network, footpaths, footpath steepness, buildings, CLUE, trees/canopy, POIs, transit, events, microclimate, DSM, point cloud | CC BY 4.0 | Train | "Source: City of Melbourne Open Data" |
| CoM `bom-weather-at-olympic-park` | Not stated (BoM data) | Validate | BoM copyright; check BoM terms before any reuse |
| OpenStreetMap | ODbL 1.0 | Train | "© OpenStreetMap contributors". **Share-alike** if we distribute a derived database (e.g. a published graph). Predictions are produced works: attribution only. |
| ESA WorldCover 2021 | CC BY 4.0 | Train | "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021)" |
| GA 5 m LiDAR DEM | CC BY 4.0 (catalog page) | Train | "© Commonwealth of Australia (Geoscience Australia)". Confirm against the licence document. |
| Copernicus GLO-30 | Copernicus DEM licence (free incl. commercial) | Train (fallback) | "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus 2014–2018, provided under COPERNICUS by the EU and ESA" |
| SRTM 1″ | Public domain | Train (fallback) | Credit NASA/USGS |
| Open-Meteo archive (ERA5) | Data CC BY 4.0; **free API tier non-commercial** | Train (course project) | Attribute Open-Meteo + Copernicus/ECMWF. A commercial product needs a subscription or a direct ERA5 pull from CDS. |
| Squiggle AFL games | No published licence (public fixture facts) | Train, low risk | Credit Squiggle; don't redistribute bulk dumps |
| Python `holidays` package (public holidays) | MIT (code); dates are facts | Train | — |
| Vic Government Important Dates | CC BY 4.0 | Validate (cross-check) | "© State of Victoria" |
| Vic school term dates (vic.gov.au) | CC BY 4.0 | Train | "© Copyright State Government of Victoria" |
| University of Melbourne semester dates | Facts; website terms forbid scraping (entered by hand) | Train | Cite the archived pages |
| Northeastern academic calendar | Facts (registrar PDF) | Train (demo) | — |
| AFL Tables attendance | No licence, no robots.txt (facts) | Train, low risk | Credit AFL Tables; don't redistribute the pages |
| USGS 3DEP (Boston) | Public domain | Train | Credit USGS |
| NAIP (Planetary Computer) | Public domain (USDA FSA) | Train | Credit USDA FSA NAIP |
| City of Boston buildings with roof breaks | PDDL 1.0 | Train | — |
| NAIP | Public domain | Train | — |
| MassGIS imagery (2025 orthos) / structures | Public domain: "the data are in the public domain and therefore can be used by anyone for any purpose" (MassGIS FAQ, confirmed 2026-10-04) | Train | "MassGIS (Bureau of Geographic Information), Commonwealth of Massachusetts EOTSS" |
| Overture buildings | ODbL (mixed) | Train (fill-in only) | Same share-alike caveat as OSM |
| **FABDEM** | **CC BY-NC-SA 4.0** | **Excluded** | Non-commercial |
| **Vicmap Elevation 1 m / Greater Melbourne LiDAR 2017–18** | Restricted / CC BY-NC (ELVIS) | **Excluded** (validation only after a team decision) | — |
| **Vicmap Buildings** | Restricted (DALA) | **Excluded** | — |
| **Stanford Drone Dataset** | **CC BY-NC-SA 3.0** (official page). Applies to any copy: the Kaggle mirrors' CC BY-SA / CC0 labels are wrong and can't relicense it | **Validate + demo** (never in a shipped model; NC + share-alike, so don't distribute demo weights) | Cite Robicquet et al., ECCV 2016. Mirrors checked 2026-10-04: Kaggle `aryashah2k/stanford-drone-dataset` and `brendanalvey/stanford-drone-dataset` both complete (8 scenes, 60 videos, identical annotation files) |
| NYC DOT pedestrian counts (bi-annual `cqsj-cfgu`; automated `ct66-47at`) | NYC Open Data Law (Admin Code §23-502(d)): published "without any registration requirement, license requirement or restrictions on their use"; source and version must be named and changes described if required | **Validate only** (cross-city transfer test; training on it would void the test) | Credit "NYC Department of Transportation, via NYC Open Data" |
