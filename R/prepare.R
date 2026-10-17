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
    puc      = rp(file.path(CACHE_DIR, "puc_hourly.parquet")),
    era5_h   = rp(file.path(CACHE_DIR, "era5_hourly.parquet"))
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
  puc <- puc_daily(cache$puc)[, .(puc_tmean = mean(tmean), puc_n = .N), by = date]
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
    ),
    micro = build_micro(cache)
  )
}

# Local-day PUC means, kept only for days with >= 20 of 24 hours observed.
puc_daily <- function(puc) {
  if (is.null(puc) || !nrow(puc)) {
    return(data.table(station_id = integer(), date = as.Date(character()), tmean = numeric()))
  }
  p <- copy(puc)[, date := as.Date(time, tz = TZ_LOCAL)]
  p[, .(tmean = mean(temp), hours = .N), by = .(station_id, date)][hours >= 20L, !"hours"]
}

# Microclimate payload: hourly PUC vs ERA5-Land at each PUC station.
# Temperatures are shipped as integer tenths of a degree C.
build_micro <- function(cache) {
  if (is.null(cache$era5_h) || !nrow(cache$era5_h)) return(NULL)
  h <- cache$era5_h[cache$puc, on = .(station_id, time), nomatch = NULL]
  h <- h[n >= 30L & !is.na(era5)]  # >= ~20 min of readings in the hour
  lt <- as.POSIXlt(h$time, tz = TZ_LOCAL)
  h[, `:=`(
    hour  = lt$hour,
    month = lt$mon + 1L,
    day   = as.integer(as.Date(time, tz = TZ_LOCAL))  # sampled unit, used as bootstrap block
  )]
  st <- cache$stations[station_id %in% h$station_id,
                       .(station_id, station_name, station_lat, station_lon)]
  st <- unique(h[, .(station_id, elevation)], by = "station_id")[st, on = "station_id"]
  st[, n_days := h[, uniqueN(day), by = station_id][st, on = "station_id", V1]]
  setorder(st, station_id)
  st[, idx := .I - 1L]
  h <- st[, .(station_id, idx)][h, on = "station_id"]
  setorder(h, idx, time)
  poly <- forsyth_polygon()
  list(
    stations = st[, .(idx, station_id, name = station_name, lat = station_lat,
                      lon = station_lon, elevation, n_days)],
    hourly = list(
      s     = h$idx,
      day   = h$day,
      hour  = h$hour,
      month = h$month,
      puc   = as.integer(round(10 * h$temp)),
      era5  = as.integer(round(10 * h$era5))
    ),
    outline = list(lon = round(poly$lon, 4), lat = round(poly$lat, 4))
  )
}
