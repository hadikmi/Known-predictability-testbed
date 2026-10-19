"""Registry of the external datasets used in Part B, with provenance.

Every entry records where the data come from, the access terms, and the
citation to use. ``docs/data.md`` is generated from this registry, so the
documentation and the code cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSource:
    key: str
    name: str
    role: str
    urls: tuple[str, ...]
    variable: str
    terms: str
    citation: str
    notes: str = ""


ERSST_V5 = DataSource(
    key="ersstv5",
    name="NOAA Extended Reconstructed Sea Surface Temperature, version 5 (ERSSTv5)",
    role="Observed predictors: tropical Pacific SST anomalies (monthly, 2 deg, 1854-present)",
    urls=("https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.ersst.v5/sst.mnmean.nc",),
    variable="sst",
    terms="NOAA data; open for public use. Attribution requested.",
    citation=(
        "Huang, B., et al. (2017): Extended Reconstructed Sea Surface "
        "Temperature, Version 5 (ERSSTv5): Upgrades, validations, and intercomparisons. "
        "J. Climate, 30, 8179-8205, doi:10.1175/JCLI-D-16-0836.1. "
        "Data provided by NOAA PSL, Boulder, Colorado, USA, from https://psl.noaa.gov."
    ),
    notes="Distributed by NOAA PSL as one file of monthly means on a 2 x 2 degree grid, longitudes 0-358 E.",
)

NCLIMGRID = DataSource(
    key="nclimgrid",
    name="NOAA Monthly U.S. Climate Gridded Dataset (nClimGrid), precipitation",
    role="Observed target: cool-season precipitation over the U.S. Southwest box (1895-present)",
    urls=(
        "https://www.ncei.noaa.gov/data/nclimgrid-monthly/access/nclimgrid_prcp.nc",
        "https://www.ncei.noaa.gov/pub/data/nidis/indices/nclimgrid-monthly/base-files/nclimgrid_prcp.nc",
    ),
    variable="prcp",
    terms=(
        "NOAA data are open to the public. Attribution is requested; do not imply NOAA endorsement, "
        "and do not present modified data as original NOAA data."
    ),
    citation=(
        "Vose, R. S., et al. (2014): Improved historical temperature and precipitation time series "
        "for U.S. climate divisions. J. Appl. Meteor. Climatol., 53, 1232-1251, "
        "doi:10.1175/JAMC-D-13-0248.1. Data: NOAA NCEI, NOAA Monthly U.S. Climate Gridded Dataset "
        "(NClimGrid), version 1, doi:10.7289/V5SX6B56."
    ),
    notes=(
        "About 5 km (1/24 deg) grid over CONUS; the full file is several GB. Only a regional mean is "
        "kept after download. Check the units attribute (monthly totals in mm)."
    ),
)

CESM2_LENS = DataSource(
    key="cesm2lens",
    name="CESM2 Large Ensemble (CESM2-LE), cloud-optimised Zarr on AWS",
    role="Training data: surface temperature (TS) and precipitation, historical + SSP3-7.0, 50 members",
    urls=(
        "s3://ncar-cesm2-lens (anonymous access, us-west-2)",
        "https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json (intake-esm catalog)",
    ),
    variable="TS, PRECC + PRECL (or PRECT)",
    terms="UCAR terms of use for data: https://www.ucar.edu/terms-of-use/data",
    citation=(
        "Rodgers, K. B., et al. (2021): Ubiquity of human-induced changes in climate variability. Earth Syst. Dynam., 12, 1393-1411, "
        "doi:10.5194/esd-12-1393-2021. Accessed from https://registry.opendata.aws/ncar-cesm2-lens."
    ),
    notes=(
        "TS (surface temperature) on the atmosphere grid is used instead of ocean-model SST to avoid "
        "regridding the curvilinear ocean grid; over the ice-free tropical ocean TS is the SST seen by "
        "the atmosphere. Monthly time stamps in CESM output mark the END of the averaging interval; "
        "they are replaced by interval midpoints (knownpred.data.preprocess.fix_cesm_time)."
    ),
)

CESM2_SMYLE = DataSource(
    key="smyle",
    name="CESM2 Seasonal-to-Multiyear Large Ensemble (SMYLE), November starts, precipitation",
    role="Perfect-model truth for real years: 20-member hindcasts from the observed 1 November state, 1970-2020",
    urls=(
        "https://thredds.rda.ucar.edu/thredds/fileServer/files/d010074/data/CESM2-SMYLE_NOV.PRE.1970-2020.nc",
        "https://tds.gdex.ucar.edu/thredds/fileServer/files/d010074/data/CESM2-SMYLE_NOV.PRE.1970-2020.nc",
    ),
    variable="PRE (mm/day)",
    terms="CC BY 4.0 (NCAR GDEX).",
    citation=(
        "Yeager, S. G., et al. (2022): The Seasonal-to-Multiyear Large Ensemble (SMYLE) prediction "
        "system using the Community Earth System Model version 2. Geosci. Model Dev., 15, 6451-6493, "
        "doi:10.5194/gmd-15-6451-2022. Data: NCAR GDEX, CESM2 SMYLE, doi:10.26024/PWMA-RE41; "
        "November-start processed collection, GDEX dataset d010074."
    ),
    notes=(
        "Only the HDF5 chunks covering January-March and the target box are read through HTTP range "
        "requests (OPeNDAP fails on this server for 64-bit integer coordinates). The regional target "
        "per year and member is cached in data/raw/smyle/smyle_PRE_target.nc with source URL and access date."
    ),
)

SOURCES = {s.key: s for s in (ERSST_V5, NCLIMGRID, CESM2_LENS, CESM2_SMYLE)}


def data_markdown() -> str:
    """Render the registry as the Markdown used in docs/data.md."""
    lines = ["| Dataset | Role | Variable | Access terms |", "|---|---|---|---|"]
    for s in SOURCES.values():
        lines.append(f"| {s.name} | {s.role} | `{s.variable}` | {s.terms} |")
    lines += ["", "### Citations", ""]
    for s in SOURCES.values():
        lines.append(f"- **{s.name}.** {s.citation}")
    lines += ["", "### Access points", ""]
    for s in SOURCES.values():
        for u in s.urls:
            lines.append(f"- {s.key}: `{u}`")
    lines += ["", "### Processing notes", ""]
    for s in SOURCES.values():
        if s.notes:
            lines.append(f"- **{s.key}**: {s.notes}")
    return "\n".join(lines) + "\n"
