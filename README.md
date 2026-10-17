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
quarto render dashboard.qmd
```

Required R packages: `birdweatheR` (GitHub, `BrentPease1/birdweatheR`), `data.table`, `nanoparquet`, `jsonlite`, `maps`, `ggplot2`, `knitr`, `rmarkdown`.
The dashboard loads Observable Plot and d3 from cdn.jsdelivr.net at view time, as all Quarto OJS pages do.

## Cache design

Detections are cached as server-side daily species counts per station, not raw detections.
One row is `station_id` (int32), `date` (Date), `species_id` (int32), `count` (int32).
A raw detection row from `get_detections()` has 21 mostly-string columns, and pulling raw rows runs at 250 per request.
Files are zstd-compressed Parquet written with `nanoparquet`, which has no compiled dependencies beyond its own sources.
Counts are partitioned by year in `data/cache/counts/<year>.parquet`.
In testing, 1.1 million count rows took 1.1 MB on disk.
PUC sensor readings (about one every 42 s) are reduced to daily min, mean, and max per station as they arrive, and raw readings are never stored.

Each update refetches the last 7 days per station and replaces those station-days whole, so late uploads and dropped species are handled.

## Definitions and assumptions

See the Methods page of the dashboard.
In short, rank is the county-wide competition rank of the trailing-window detection sum.
Normalized count is either the share of all detections or detections per active station-day.
Temperature is ERA5 daily mean, min, and max at the county centroid (Open-Meteo), with BirdWeather PUC on-board sensor daily means overlaid.
Detections measure vocal activity weighted by detectability, not abundance.
