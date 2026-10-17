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
    era5_h   = rp(file.path(CACHE_DIR, "era5_hourly.parquet")),
    seasonal = rp(file.path(CACHE_DIR, "seasonal.parquet"))  # scripts/seasonal.py
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
  puc <- puc_daily(cache$puc, cache$era5_h)[, .(
    puc_tmean  = mean(tmean),
    puc_rh     = mean(rh, na.rm = TRUE),
    puc_rh_adj = mean(rh_adj, na.rm = TRUE),
    puc_n      = .N
  ), by = date]
  weather <- puc[weather, on = "date"]
  for (v in c("rhmin", "rhmean", "rhmax")) if (!v %in% names(weather)) weather[, (v) := NA_real_]

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
      puc_tmean = round(weather$puc_tmean, 1),
      rhmin  = round(weather$rhmin),
      rhmean = round(weather$rhmean),
      rhmax  = round(weather$rhmax),
      puc_rh     = round(weather$puc_rh),
      puc_rh_adj = round(weather$puc_rh_adj)
    ),
    micro = build_micro(cache),
    seasonal = build_seasonal(cache$seasonal, top)
  )
}

# Annual seasonal component from scripts/seasonal.py, for the shipped species.
# Columnar like `series`: species-major, one row per (species, grid day), with
# NA for species the script did not fit.
build_seasonal <- function(seas, top) {
  if (is.null(seas) || !nrow(seas)) return(NULL)
  seas <- copy(seas)[, norm := as.character(norm)]
  doy <- sort(unique(seas$doy))
  out <- list(doy = doy)
  for (nm in c("share", "per_station")) {
    g <- CJ(idx = top$idx, doy = doy)
    g[top, on = "idx", species_id := i.species_id]
    g <- seas[norm == nm][g, on = .(species_id, doy)]
    setorder(g, idx, doy)
    out[[nm]] <- list(est = round(g$est, 3), lo = round(g$lo, 3), hi = round(g$hi, 3))
  }
  out
}

# Saturation vapor pressure over water (hPa), Magnus form (Alduchov & Eskridge
# coefficients), adequate for -40 to 50 deg C.
svp <- function(t) 6.1094 * exp(17.625 * t / (t + 243.04))

# Dew point (deg C) from air temperature (deg C) and RH (%), inverse Magnus.
dew_point <- function(t, rh) {
  g <- log(rh / 100) + 17.625 * t / (t + 243.04)
  243.04 * g / (17.625 - g)
}

# Local-day PUC means, kept only for days with >= 20 of 24 hours observed.
# rh is the sensor reading. rh_adj re-expresses the same vapor pressure at the
# ERA5-Land air temperature for that station-hour: the enclosure warms the air
# without adding water, so the sensor's dew point carries over but its RH does
# not.
puc_daily <- function(puc, era5_h = NULL) {
  if (is.null(puc) || !nrow(puc)) {
    return(data.table(station_id = integer(), date = as.Date(character()),
                      tmean = numeric(), rh = numeric(), rh_adj = numeric()))
  }
  p <- copy(puc)[, date := as.Date(time, tz = TZ_LOCAL)]
  p[, rh := fifelse(humidity > 0 & humidity <= 100, humidity, NA_real_)]
  if (!is.null(era5_h) && nrow(era5_h)) {
    p <- era5_h[, .(station_id, time, era5)][p, on = .(station_id, time)]
    p[, rh_adj := pmin(100, rh * svp(temp) / svp(era5))]
  } else {
    p[, rh_adj := NA_real_]
  }
  p[, .(tmean = mean(temp), rh = mean(rh, na.rm = TRUE), rh_adj = mean(rh_adj, na.rm = TRUE),
        hours = .N), by = .(station_id, date)][hours >= 20L, !"hours"]
}

# Microclimate payload: hourly PUC vs ERA5-Land at each PUC station.
# Temperature and dew point ship as integer tenths of a degree C, RH as
# integer tenths of a percent.
build_micro <- function(cache) {
  if (is.null(cache$era5_h) || !nrow(cache$era5_h)) return(NULL)
  h <- cache$era5_h[cache$puc, on = .(station_id, time), nomatch = NULL]
  h <- h[n >= 30L & !is.na(era5)]  # >= ~20 min of readings in the hour
  for (v in c("era5_rh", "era5_td")) if (!v %in% names(h)) h[, (v) := NA_real_]
  # Humidity: dew point is unchanged by the warm enclosure, so it compares
  # directly; RH is re-expressed at the ERA5-Land air temperature (see puc_daily).
  h[, rh := fifelse(humidity > 0 & humidity <= 100, humidity, NA_real_)]
  h[, `:=`(puc_td = dew_point(temp, rh), puc_rha = pmin(100, rh * svp(temp) / svp(era5)))]
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
      era5  = as.integer(round(10 * h$era5)),
      puc_td  = as.integer(round(10 * h$puc_td)),
      era5_td = as.integer(round(10 * h$era5_td)),
      puc_rh  = as.integer(round(10 * h$puc_rha)),
      era5_rh = as.integer(round(10 * h$era5_rh))
    ),
    outline = list(lon = round(poly$lon, 4), lat = round(poly$lat, 4))
  )
}

# Local ecology table: top species with Wikipedia links, curated status
# (data/species_status.csv), a West Nile virus flag (CDC list of dead-bird
# species with WNV detected, 1999-2016), and an effort-normalized seasonality
# profile.
# Seasonality is each species' share of all detections in a calendar month,
# pooled over years and scaled to the species' peak month, so months with more
# stations online do not look busier.
# Names compared without case, hyphens, apostrophes, or spaces, so that
# "Black-crowned Night Heron" and "Black-crowned Night-Heron" match.
norm_name <- function(x) gsub("[^a-z]", "", tolower(x))

build_ecology <- function(cache, n = 15L, status_file = "data/species_status.csv",
                          wnv_file = "data/wnv_bird_species_cdc_1999_2016.csv") {
  cnt <- cache$counts
  top <- cnt[, .(detections = sum(count)), by = species_id][order(-detections)][seq_len(min(n, .N))]
  top <- cache$species[top, on = "species_id"]
  top[, rank := .I]
  status <- if (file.exists(status_file)) fread(status_file) else data.table(scientific_name = character())
  top <- status[top, on = "scientific_name"]
  top[is.na(status), `:=`(status = "Not classified", note = "", source = "")]
  wnv <- if (file.exists(wnv_file)) fread(wnv_file)$common_name else character()
  top[, wnv := norm_name(common_name) %in% norm_name(wnv)]

  m <- cnt[, .(n = sum(count)), by = .(species_id, month = month(date))]
  m[, share := n / sum(n), by = month]
  m <- m[species_id %in% top$species_id]
  grid <- CJ(species_id = top$species_id, month = 1:12)
  m <- m[grid, on = .(species_id, month)][is.na(share), share := 0]
  bars <- c("▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")
  spark <- m[order(month), .(seasonality = {
    r <- share / max(share)
    paste(bars[pmax(1L, ceiling(r * length(bars)))], collapse = "")
  }), by = species_id]
  top <- spark[top, on = "species_id"]
  setorder(top, rank)
  top[, .(rank, common_name, scientific_name, detections, status, note, source, seasonality, wnv)]
}
