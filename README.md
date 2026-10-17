# Forsyth County birds dashboard

Quarto dashboard of BirdWeather detections for stations in Forsyth County, NC.
It shows species rank over time, normalized detection counts, and daily temperature.

## Layout

| Path | Purpose |
|---|---|
| `R/config.R` | Seed (1834), county polygon, bbox, centroid, paths |
| `R/cache.R` | API pulls and incremental Parquet cache |
| `R/prepare.R` | Builds the compact browser payload (ranks, daily totals, temperature) |
| `scripts/update_cache.R` | Refreshes the cache |
| `dashboard.qmd` | Dashboard (Observable Plot via Quarto OJS) |
| `data/cache/` | Parquet cache |

## Run

```bash
Rscript scripts/update_cache.R   # first run pulls full history; later runs fetch the last 7 days
uv sync                          # installs the pinned Quarto CLI (quarto-cli on PyPI)
uv run quarto render dashboard.qmd
```

Quarto is pinned in `pyproject.toml` / `uv.lock` and run through uv.
Required R packages: `birdweatheR` (GitHub, `BrentPease1/birdweatheR`), `data.table`, `nanoparquet`, `jsonlite`, `maps`, `ggplot2`, `knitr`, `rmarkdown`.
The dashboard loads Observable Plot and d3 from cdn.jsdelivr.net at view time, as all Quarto OJS pages do.

## Cache design

Detections are cached as server-side daily species counts per station, not raw detections.
One row is `station_id` (int32), `date` (Date), `species_id` (int32), `count` (int32).
A raw detection row from `get_detections()` has 21 mostly-string columns, and pulling raw rows runs at 250 per request.
Files are zstd-compressed Parquet written with `nanoparquet`, which has no compiled dependencies beyond its own sources.
Counts are partitioned by year in `data/cache/counts/<year>.parquet`.
In testing, 1.1 million count rows took 1.1 MB on disk.
PUC sensor readings (about one every 42 s) are pulled on a stratified random subsample of days, 4 UTC days per calendar month by default (`BW_PUC_DAYS_PER_MONTH`, `Inf` for all days).
The sample is seeded from (1834, year, month), so it is stable across runs and shared by all stations.
Each sampled day is pulled in full and reduced to hourly means before writing; raw readings are never stored.
ERA5-Land hourly temperature at each PUC station's coordinates comes from the Open-Meteo archive for the same hours.

Each update refetches the last 7 days per station and replaces those station-days whole, so late uploads and dropped species are handled.

## Definitions and assumptions

See the Methods page of the dashboard.
In short, rank is the county-wide competition rank of the trailing-window detection sum.
Normalized count is either the share of all detections or detections per active station-day.
Temperature is ERA5 daily mean, min, and max at the county centroid (Open-Meteo), with BirdWeather PUC on-board sensor daily means overlaid.
The Microclimate page compares each PUC station with ERA5-Land at its own location, with day-block bootstrap intervals, the delta by hour of day, and between-station spread compared with ERA5-Land spread.
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
