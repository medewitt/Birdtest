# Forsyth County birds dashboard

Quarto dashboard of BirdWeather detections, temperature data, and other environmental data for stations in Forsyth County, NC.
It shows species rank over time, normalized detection counts, and daily temperature.

## Layout

| Path | Purpose |
|---|---|
| `R/config.R` | Seed (1834), county polygon, bbox, centroid, paths |
| `R/cache.R` | API pulls and incremental Parquet cache |
| `R/prepare.R` | Builds the compact browser payload (ranks, daily totals, temperature) |
| `scripts/update_cache.R` | Refreshes the cache |
| `scripts/seasonal.py` | Fits each shipped species' annual seasonal component (harmonic regression, block bootstrap) to `data/cache/seasonal.parquet` |
| `scripts/site_compare.py` | NumPyro models comparing PUC sites: overall detection rate, species share (constant and month-varying site effects, compared by PSIS-LOO), and day-to-day weather response; writes `output/site_compare/` |
| `scripts/phenology.py` | NumPyro hurdle model per species (presence and count splines, site offsets): probabilistic spring onset (7 days at 2x the winter mean) and evidence on arrival vs song; writes `output/phenology/` |
| `scripts/terrain.py` | One-off: USGS 3DEP elevation (3 arc-second) to TPI, D8 drainage area, derived streams, and height above nearest drainage; writes `data/terrain.parquet` |
| `scripts/gp_weekly.py` | NumPyro Gaussian process on weekly CoCoRaHS rain and PUC night temperature delta: weekly county surfaces, persistent spatial field, and per-site offsets; writes `data/cache/gp_*.parquet` |
| `data/terrain.parquet` | Static terrain and drainage layer (0.0025 deg cells) from `scripts/terrain.py` |
| `dashboard.qmd` | Dashboard (Observable Plot via Quarto OJS) |
| `data/species_status.csv` | Curated residency status, note, and source DOI for the Local ecology table |
| `data/wnv_bird_species_cdc_1999_2016.csv` | CDC list of dead-bird species with West Nile virus detected, US 1999-2016 (332 species) |
| `assets/mosquito.png` | Icon marking WNV-listed species in the Local ecology table |
| `assets/footer.html` | IDEAS footer |
| `assets/head.html`, `assets/fonts/` | Self-hosted Nunito Sans (SIL OFL 1.1) as per IDEAS lab convention |
| `dashboard.scss` | Theme: IDEAS colors (ink, old gold, dark navbar), phone layout, table styles |
| `data/cache/` | Parquet cache |

## To Build

```bash
Rscript scripts/update_cache.R   # first run pulls full history; later runs fetch the last 7 days
uv sync                          # installs the pinned Quarto CLI (quarto-cli on PyPI)
uv run python scripts/seasonal.py       # seasonal component per species (about 15 s)
uv run --group analysis python scripts/gp_weekly.py   # weekly GP surfaces; skips if no new complete week
uv run quarto render dashboard.qmd
```

The site comparison is a separate analysis, not part of the dashboard build:

```bash
uv run --group analysis python scripts/site_compare.py   # about 12 min on 4 CPU cores
uv run --group analysis python scripts/phenology.py      # about 85 min on 4 CPU cores
```

Quarto and the Python dependencies (numpy, pyarrow) are pinned in `pyproject.toml` / `uv.lock` and run through uv.
Required R packages: `birdweatheR` (GitHub, `BrentPease1/birdweatheR`), `data.table`, `nanoparquet`, `jsonlite`, `maps`, `ggplot2`, `knitr`, `rmarkdown`.
The dashboard loads Observable Plot and d3 from cdn.jsdelivr.net at view time, as all Quarto OJS pages do.

## Cache design

Detections are cached as server-side daily species counts per station, not raw detections.
One row is `station_id` (int32), `date` (Date), `species_id` (int32), `count` (int32).
A raw detection row from `get_detections()` has 21 mostly-string columns, and pulling raw rows runs at 250 per request.
Files are zstd-compressed Parquet written with `nanoparquet`, which has no compiled dependencies beyond its own sources.
Counts are partitioned by year in `data/cache/counts/<year>.parquet`.
In testing, 1.1 million count rows took 1.1 MB on disk.
PUC sensor readings (about one every 42 s) are pulled on a stratified random subsample of days, 4 local calendar days per month by default (`BW_PUC_DAYS_PER_MONTH`, `Inf` for all days).
The sample is seeded from (1834, year, month), so it is stable across runs and shared by all stations.
Each sampled day is pulled in full and reduced to hourly means before writing; raw readings are never stored.
[CoCoRaHS](https://www.cocorahs.org) precipitation is stored as one row per gauge-day report (`cocorahs_daily.parquet`, inches as reported, with trace and accumulation codes) plus a gauge table (`cocorahs_stations.parquet`).
ERA5-Land hourly temperature at each PUC station's coordinates comes from the Open-Meteo archive for the same hours.

Each update refetches the last 7 days per station and replaces those station-days whole, so late uploads and dropped species are handled.

## Definitions and assumptions

See the Methods page of the dashboard.
As per the current methodology, species rank is the county-wide competition rank of the trailing-window (i.e., lag) detection sum.
Normalized count is either the share of all detections or detections per active station-day.
Temperature and relative humidity are ERA5 daily mean, min, and max at the county centroid (Open-Meteo), with BirdWeather PUC on-board sensor daily means overlaid; PUC humidity is also shown re-expressed at outdoor (ERA5-Land) temperature, since the warm enclosure makes the sensor read dry.
The Microclimate page compares each PUC station with ERA5-Land at its own location for air temperature, dew point, or relative humidity, with day-block bootstrap intervals, the delta by hour of day, and between-station spread compared with ERA5-Land spread.
The Microclimate page also carries weekly Gaussian process surfaces for CoCoRaHS rain and PUC night temperature delta, with site offsets compared against terrain (`data/terrain.parquet`, rebuilt only by hand with `uv run --group analysis python scripts/terrain.py`).
Precipitation is the mean of CoCoRaHS gauge reports in the county (RCC-ACIS `MultiStnData`, FIPS 37067), each shifted to the day before its morning observation date; the full history is refetched each run because observers report late.
Detections measure vocal activity weighted by detectability, not abundance.

## Automation

`.github/workflows/dashboard.yml` runs daily at 10:17 UTC and on manual dispatch.
It updates the cache, renders the dashboard, and deploys it to GitHub Pages.
Pushes to `main` and pull requests only render from the stored cache and make no API calls.

The cache is kept on an orphan `cache` branch, checked out as a worktree at `data/cache/`.
Each update force-pushes a single parentless commit, so the Parquet files never accumulate in git history.
To work with it locally, run `git worktree add data/cache cache`.
The manual dispatch has a `full_refresh` option that discards the cache and pulls full history.

One-time setup: in the repository settings, set Pages to deploy from GitHub Actions, and make sure Actions has read and write permission for contents.

The Local ecology page lists the top 15 species with Wikipedia links, residency status for the NC Piedmont (Resident, Migratory, or Both) from `data/species_status.csv`, and an effort-normalized monthly seasonality profile computed from the detections.
A mosquito icon marks species on the CDC list of dead-bird species in which West Nile virus has been detected (US, 1999-2016), matched by common name.
Species that enter the top 15 without a row in the CSV show as "Not classified" until a row is added.
