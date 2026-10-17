# Turn the cache into the compact payload the dashboard ships to the browser.
#
# Definitions
#   count      county total of BirdNET detections for a species on a day,
#              summed over stations (server-side daily counts, station
#              confidence thresholds apply; no extra filter here).
#   rank       competition ("min") rank of the trailing w-day sum among all
#              species detected in that window (1 = most detected).
#   share      trailing w-day species sum / trailing w-day all-species sum.
#   per_station trailing w-day species sum / trailing w-day active
#              station-days. A station is active on a day it reports any
#              detection; silent-but-online stations are not counted.

library(data.table)

WINDOWS <- c(1L, 7L, 28L)
N_SHIP  <- 40L  # species shipped to the browser (top by all-time count)

read_cache <- function() {
  f <- list.files(file.path(CACHE_DIR, "counts"), "\\.parquet$", full.names = TRUE)
  rp <- function(p) if (file.exists(p)) as.data.table(nanoparquet::read_parquet(p))
  list(
    counts   = rbindlist(lapply(f, rp)),
    species  = rp(file.path(CACHE_DIR, "species.parquet")),
    stations = rp(file.path(CACHE_DIR, "stations.parquet")),
    weather  = rp(file.path(CACHE_DIR, "weather_era5.parquet")),
    puc      = rp(file.path(CACHE_DIR, "puc_temp.parquet"))
  )
}

build_payload <- function(cache = read_cache()) {
  cnt <- cache$counts[, .(count = sum(count)), by = .(date, species_id)]
  days <- seq(min(cnt$date), max(cnt$date), by = "day")

  daily <- cache$counts[, .(
    total      = sum(count),
    n_stations = uniqueN(station_id)
  ), by = date]
  daily <- daily[data.table(date = days), on = "date"]
  setnafill(daily, fill = 0L, cols = c("total", "n_stations"))

  # Dense species x day grid over all species, needed for honest ranks.
  grid <- CJ(species_id = unique(cnt$species_id), date = days)
  grid <- cnt[grid, on = .(species_id, date)]
  setnafill(grid, fill = 0L, cols = "count")
  setkey(grid, species_id, date)

  for (w in WINDOWS) {
    s <- paste0("s", w)
    r <- paste0("r", w)
    grid[, (s) := frollsum(count, w, align = "right"), by = species_id]
    grid[get(s) > 0, (r) := frank(-get(s), ties.method = "min"), by = date]
  }

  top <- grid[, .(n = sum(count)), by = species_id][order(-n)][seq_len(min(N_SHIP, .N))]
  top <- cache$species[top, on = "species_id"]
  # Charts key on common_name, so it must be present and unique.
  miss <- top[is.na(common_name) | common_name == "", species_id]
  if (length(miss)) {
    warning("No common name for species_id ", paste(miss, collapse = ", "),
            "; using scientific name. Re-run scripts/update_cache.R.", call. = FALSE)
    top[species_id %in% miss, common_name := fcoalesce(scientific_name, paste("species_id", species_id))]
  }
  top[duplicated(common_name) | duplicated(common_name, fromLast = TRUE),
      common_name := paste0(common_name, " (", scientific_name, ")")]
  top[, idx := .I - 1L]

  ship <- grid[top[, .(species_id, idx)], on = "species_id"]
  setorder(ship, idx, date)
  rank_cols <- paste0("r", WINDOWS)

  weather <- cache$weather[data.table(date = days), on = "date"]
  puc <- if (!is.null(cache$puc) && nrow(cache$puc)) {
    cache$puc[n >= 500, .(puc_tmean = mean(tmean), puc_n = .N), by = date]
  } else data.table(date = as.Date(character()), puc_tmean = numeric(), puc_n = integer())
  weather <- puc[weather, on = "date"]

  list(
    meta = list(
      windows   = WINDOWS,
      first_day = as.integer(days[1]),
      n_days    = length(days),
      n_species_total = uniqueN(cnt$species_id),
      n_stations = nrow(cache$stations),
      n_puc     = sum(cache$stations$station_type == "puc"),
      data_through = format(max(days)),
      rendered  = format(Sys.time(), "%Y-%m-%d %H:%M %Z")
    ),
    species = top[, .(idx, common_name, scientific_name, total = n)],
    # Columnar, species-major, one row per (species, day). Day index is
    # implicit: row i belongs to day i %% n_days.
    series = c(list(count = ship$count), lapply(ship[, ..rank_cols], function(x) x)),
    daily = list(
      total      = daily$total,
      n_stations = daily$n_stations,
      tmin  = round(weather$tmin, 1),
      tmean = round(weather$tmean, 1),
      tmax  = round(weather$tmax, 1),
      puc_tmean = round(weather$puc_tmean, 1)
    )
  )
}
