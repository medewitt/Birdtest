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
puc      <- update_puc_temp(stations)

sz <- sum(file.size(list.files(CACHE_DIR, recursive = TRUE, full.names = TRUE)))
message(sprintf(
  "Cache: %s count rows, %d species, %d stations (%d PUC), %.2f MB on disk",
  format(nrow(counts), big.mark = ","), nrow(species), nrow(stations),
  sum(stations$station_type == "puc"), sz / 2^20
))
