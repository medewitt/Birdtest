# Refresh the Parquet cache. Run from the project root:
#   Rscript scripts/update_cache.R
# First run pulls full history; later runs fetch only the last few days.

source("R/config.R")
source("R/cache.R")

connect_birdweather()

stations <- update_stations()
counts   <- update_counts(stations)
species  <- update_species(counts)
weather  <- update_weather()
puc      <- update_puc_hourly(stations, counts)
era5_h   <- update_era5_hourly(stations, puc)
# Precipitation is a side variable; an ACIS outage should not block the bird update.
precip   <- tryCatch(update_cocorahs(), error = function(e) {
  message("CoCoRaHS update skipped: ", conditionMessage(e))
  NULL
})

sz <- sum(file.size(list.files(CACHE_DIR, recursive = TRUE, full.names = TRUE)))
message(sprintf(
  "Cache: %s count rows, %d species, %d stations (%d PUC), %.2f MB on disk",
  format(nrow(counts), big.mark = ","), nrow(species), nrow(stations),
  sum(stations$station_type == "puc"), sz / 2^20
))
