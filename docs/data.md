# Data sources (Part B)

Generated from `src/knownpred/data/sources.py` (`python -m knownpred.cli data-docs`). Part A uses no external data: every number comes from the toy system simulated by this repository.

Raw data are never committed. After download, `data/raw/MANIFEST.json` records the URL, access date, size and SHA-256 checksum of every file used.

| Dataset | Role | Variable | Access terms |
|---|---|---|---|
| NOAA Extended Reconstructed Sea Surface Temperature, version 5 (ERSSTv5) | Observed predictors: tropical Pacific SST anomalies (monthly, 2 deg, 1854-present) | `sst` | NOAA data; open for public use. Attribution requested. |
| NOAA Monthly U.S. Climate Gridded Dataset (nClimGrid), precipitation | Observed target: cool-season precipitation over the U.S. Southwest box (1895-present) | `prcp` | NOAA data are open to the public. Attribution is requested; do not imply NOAA endorsement, and do not present modified data as original NOAA data. |
| CESM2 Large Ensemble (CESM2-LE), cloud-optimised Zarr on AWS | Training data: surface temperature (TS) and precipitation, historical + SSP3-7.0, 50 members | `TS, PRECC + PRECL (or PRECT)` | UCAR terms of use for data: https://www.ucar.edu/terms-of-use/data |

### Citations

- **NOAA Extended Reconstructed Sea Surface Temperature, version 5 (ERSSTv5).** Huang, B., et al. (2017): Extended Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5): Upgrades, validations, and intercomparisons. J. Climate, 30, 8179-8205, doi:10.1175/JCLI-D-16-0836.1. Data provided by NOAA PSL, Boulder, Colorado, USA, from https://psl.noaa.gov.
- **NOAA Monthly U.S. Climate Gridded Dataset (nClimGrid), precipitation.** Vose, R. S., et al. (2014): Improved historical temperature and precipitation time series for U.S. climate divisions. J. Appl. Meteor. Climatol., 53, 1232-1251, doi:10.1175/JAMC-D-13-0248.1. Data: NOAA NCEI, NOAA Monthly U.S. Climate Gridded Dataset (NClimGrid), version 1, doi:10.7289/V5SX6B56.
- **CESM2 Large Ensemble (CESM2-LE), cloud-optimised Zarr on AWS.** Rodgers, K. B., et al. (2021): Ubiquity of human-induced changes in climate variability. Earth Syst. Dynam., 12, 1393-1411, doi:10.5194/esd-12-1393-2021. Accessed from https://registry.opendata.aws/ncar-cesm2-lens.

### Access points

- ersstv5: `https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.ersst.v5/sst.mnmean.nc`
- nclimgrid: `https://www.ncei.noaa.gov/data/nclimgrid-monthly/access/nclimgrid_prcp.nc`
- nclimgrid: `https://www.ncei.noaa.gov/pub/data/nidis/indices/nclimgrid-monthly/base-files/nclimgrid_prcp.nc`
- cesm2lens: `s3://ncar-cesm2-lens (anonymous access, us-west-2)`
- cesm2lens: `https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/data/aws-cesm2-le.json (intake-esm catalog)`

### Processing notes

- **ersstv5**: Distributed by NOAA PSL as one file of monthly means on a 2 x 2 degree grid, longitudes 0-358 E.
- **nclimgrid**: About 5 km (1/24 deg) grid over CONUS; the full file is several GB. Only a regional mean is kept after download. Check the units attribute (monthly totals in mm).
- **cesm2lens**: TS (surface temperature) on the atmosphere grid is used instead of ocean-model SST to avoid regridding the curvilinear ocean grid; over the ice-free tropical ocean TS is the SST seen by the atmosphere. Monthly time stamps in CESM output mark the END of the averaging interval; they are replaced by interval midpoints (knownpred.data.preprocess.fix_cesm_time).
