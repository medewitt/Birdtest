# Fetch BirdWeather + weather data and maintain an incremental Parquet cache.
#
# Cache layout (all zstd-compressed Parquet via nanoparquet):
#   stations.parquet           one row per Forsyth station
#   counts/<year>.parquet      station_id, date, species_id, count (all int32/Date)
#   progress.parquet           station_id, fetched_through
#   species.parquet            species_id, common_name, scientific_name
#   weather_era5.parquet       date, tmin, tmean, tmax (deg C, Open-Meteo ERA5)
#   puc_hourly.parquet         station_id, time (UTC hour), temp, humidity, n
#   puc_days_done.parquet      station_id, date (sampled local days fetched), version
#   era5_hourly.parquet        station_id, time, era5 (deg C, ERA5-Land), elevation
#
# Detections are cached as server-side daily species counts, not raw rows.
# A raw detection is ~21 columns of strings; a daily count is 4 integers.

library(data.table)
library(birdweatheR)

cache_path <- function(...) file.path(CACHE_DIR, ...)

write_pq <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  nanoparquet::write_parquet(x, path, compression = "zstd")
}

read_pq <- function(path) {
  if (!file.exists(path)) return(NULL)
  as.data.table(nanoparquet::read_parquet(path))
}

with_retry <- function(f, max_tries = 5) {
  for (i in seq_len(max_tries)) {
    out <- tryCatch(f(), error = function(e) e)
    if (!inherits(out, "error")) return(out)
    if (i == max_tries) stop(out)
    wait <- 2^i + runif(1)
    message("  retry ", i, " in ", round(wait, 1), "s: ", conditionMessage(out))
    Sys.sleep(wait)
  }
}

# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------
update_stations <- function() {
  bb <- forsyth_bbox()
  st <- get_stations(ne = bb$ne, sw = bb$sw)
  poly <- forsyth_polygon()
  st <- st[in_polygon(station_lon, station_lat, poly$lon, poly$lat)]
  st[, station_id := as.integer(station_id)]
  setkey(st, station_id)
  write_pq(st, cache_path("stations.parquet"))
  message(nrow(st), " stations inside Forsyth County")
  st
}

# ---------------------------------------------------------------------------
# Daily species counts per station, chunked by year, incremental
# ---------------------------------------------------------------------------
fetch_station_counts <- function(sid, from, to) {
  starts <- seq(from, to, by = "year")
  out <- lapply(starts, function(s) {
    e <- min(seq(s, by = "year", length.out = 2)[2] - 1, to)
    d <- with_retry(function() {
      get_daily_detection_counts(
        from = s, to = e, station_ids = as.character(sid), by_species = TRUE
      )
    })
    if (!nrow(d)) return(NULL)
    d[, .(
      station_id = as.integer(sid),
      date       = as.Date(date),
      species_id = as.integer(species_id),
      count      = as.integer(count)
    )]
  })
  rbindlist(out)
}

read_counts <- function() {
  f <- list.files(cache_path("counts"), "\\.parquet$", full.names = TRUE)
  rbindlist(lapply(f, read_pq))
}

# Re-fetches the last `overlap` days on every run, since the most recent day
# is partial and late uploads from stations can backfill a few days.
update_counts <- function(stations, today = Sys.Date(), overlap = 7L) {
  prog <- read_pq(cache_path("progress.parquet"))
  if (is.null(prog)) prog <- data.table(station_id = integer(), fetched_through = as.Date(character()))

  new <- vector("list", nrow(stations))
  for (i in seq_len(nrow(stations))) {
    sid <- stations$station_id[i]
    last <- prog[station_id == sid, fetched_through]
    from <- if (length(last)) last - overlap else HISTORY_START
    message("Station ", i, "/", nrow(stations), " (", sid, ") from ", from)
    new[[i]] <- fetch_station_counts(sid, from, today)
    Sys.sleep(0.5)
  }
  new <- rbindlist(new)
  if (!nrow(new)) return(invisible(read_counts()))

  # Replace each touched (station, date) wholesale so dropped species vanish.
  touched <- unique(new[, .(station_id, date)])
  for (yr in sort(unique(year(new$date)))) {
    f <- cache_path("counts", paste0(yr, ".parquet"))
    old <- read_pq(f)
    add <- new[year(date) == yr]
    if (!is.null(old)) old <- old[!touched, on = .(station_id, date)]
    yr_dt <- rbindlist(list(old, add))
    setkey(yr_dt, date, station_id, species_id)
    write_pq(yr_dt, f)
  }

  latest <- new[, .(fetched_through = max(date)), by = station_id]
  prog <- rbindlist(list(prog[!latest, on = "station_id"], latest))
  write_pq(prog, cache_path("progress.parquet"))
  invisible(read_counts())
}

# ---------------------------------------------------------------------------
# Species lookup (only IDs not already cached)
# ---------------------------------------------------------------------------
update_species <- function(counts) {
  sp <- read_pq(cache_path("species.parquet"))
  need <- setdiff(unique(counts$species_id), sp$species_id)
  if (length(need)) {
    chunks <- split(need, ceiling(seq_along(need) / 100))
    add <- rbindlist(lapply(chunks, function(ids) {
      d <- with_retry(function() get_species_info(ids))
      d[, .(species_id = as.integer(species_id), common_name, scientific_name)]
    }))
    sp <- rbindlist(list(sp, add))
    setkey(sp, species_id)
    write_pq(sp, cache_path("species.parquet"))
  }
  sp
}

# ---------------------------------------------------------------------------
# Gridded daily temperature (Open-Meteo ERA5 archive, ~5 day lag, no key)
# ---------------------------------------------------------------------------
update_weather <- function(from = HISTORY_START, to = Sys.Date()) {
  ctr <- forsyth_centroid()
  url <- sprintf(paste0(
    "https://archive-api.open-meteo.com/v1/archive?latitude=%.4f&longitude=%.4f",
    "&start_date=%s&end_date=%s&timezone=%s",
    "&daily=temperature_2m_min,temperature_2m_mean,temperature_2m_max"
  ), ctr[["lat"]], ctr[["lon"]], from, to, utils::URLencode(TZ_LOCAL, reserved = TRUE))
  j <- with_retry(function() jsonlite::fromJSON(url))
  w <- data.table(
    date  = as.Date(j$daily$time),
    tmin  = j$daily$temperature_2m_min,
    tmean = j$daily$temperature_2m_mean,
    tmax  = j$daily$temperature_2m_max
  )[!is.na(tmean)]
  write_pq(w, cache_path("weather_era5.parquet"))
  w
}

# ---------------------------------------------------------------------------
# PUC on-board temperature on a stratified random subsample of days.
#
# Each calendar month gets PUC_DAYS_PER_MONTH days drawn with a seed derived
# from (SEED, year, month), so the sample is stable across runs and shared by
# all stations (paired comparisons). The API resolves a period to the local
# (America/New_York) calendar day named in the request, so sampled days are
# local days, each pulled in full so diurnal cycles are complete. Raw readings
# (~2,000/day/station) are reduced to hourly means before anything is written.
# ---------------------------------------------------------------------------
puc_sample_days <- function(from = HISTORY_START, to = Sys.Date() - 1L,
                            k = PUC_DAYS_PER_MONTH) {
  months <- seq(as.Date(format(from, "%Y-%m-01")), to, by = "month")
  days <- lapply(months, function(m) {
    d <- seq(m, seq(m, by = "month", length.out = 2)[2] - 1L, by = "day")
    if (is.finite(k) && k < length(d)) {
      set.seed(SEED + as.integer(format(m, "%Y")) * 12L + as.integer(format(m, "%m")))
      d <- sort(sample(d, k))
    }
    d
  })
  d <- do.call(c, days)
  d[d >= from & d <= to]
}

# ISO8601 with Z or +HH:MM offset -> POSIXct UTC.
parse_ts <- function(x) {
  x <- sub("Z$", "+0000", x)
  x <- sub("([+-]\\d{2}):(\\d{2})$", "\\1\\2", x)
  x <- sub("\\.\\d+([+-])", "\\1", x)
  as.POSIXct(x, format = "%Y-%m-%dT%H:%M:%S%z", tz = "UTC")
}

# One local day of PUC environment readings (timestamp, temperature, humidity).
# birdweatheR::get_environment_data() post-filters to the UTC day, which cuts
# local 20:00-23:59 EDT from a local-day response, so this calls the same
# GraphQL query directly and filters to the local date here.
fetch_env_day <- function(sid, d) {
  q <- '
    query station($id: ID!, $period: InputDuration, $first: Int, $after: String) {
      station(id: $id) {
        sensors {
          environmentHistory(period: $period, first: $first, after: $after) {
            pageInfo { hasNextPage endCursor }
            edges { node { timestamp temperature humidity } }
          }
        }
      }
    }'
  # birdweatheR:::fetch_page_with_retry() executes query_exec$url_link.
  exec <- ghql::Query$new()$query("url_link", q)
  vars <- list(id = as.character(sid), first = 250L,
               period = list(from = paste0(d, "T00:00:00.000Z"), to = paste0(d, "T23:59:59.000Z")))
  pages <- list()
  repeat {
    res <- birdweatheR:::fetch_page_with_retry(exec, vars)
    h <- res$data$station$sensors$environmentHistory
    if (is.null(h$edges) || !length(h$edges)) break
    pages[[length(pages) + 1]] <- as.data.table(h$edges$node)
    if (!isTRUE(h$pageInfo$hasNextPage)) break
    vars$after <- h$pageInfo$endCursor
  }
  r <- rbindlist(pages, fill = TRUE)
  if (!nrow(r)) return(r)
  r[, ts := parse_ts(timestamp)]
  r[!is.na(ts) & as.Date(ts, tz = TZ_LOCAL) == d]
}

# Plausible air temperature range for Forsyth County; readings outside it are
# sensor faults or enclosure heating, not air temperature.
PUC_TEMP_RANGE <- c(-30, 50)

# Bump when the PUC cache layout or sampling unit changes; a mismatch refetches.
PUC_CACHE_VERSION <- "local-day-v2"

update_puc_hourly <- function(stations, counts = NULL, days = puc_sample_days()) {
  puc  <- stations[station_type == "puc", station_id]
  old  <- read_pq(cache_path("puc_hourly.parquet"))
  done <- read_pq(cache_path("puc_days_done.parquet"))
  # v1 caches sampled UTC days through the package filter (missing 4 h/day).
  if (is.null(done) || !identical(PUC_CACHE_VERSION, done$version[1])) {
    if (!is.null(done)) message("PUC cache format changed; refetching sampled days.")
    old  <- NULL
    done <- data.table(station_id = integer(), date = as.Date(character()), version = character())
  }
  first <- if (!is.null(counts)) counts[, .(first = min(date)), by = station_id] else NULL
  out <- list(old)
  for (sid in puc) {
    todo <- setdiff(days, done[station_id == sid, date])
    todo <- as.Date(todo, origin = "1970-01-01")
    f <- first[station_id == sid, first]
    if (length(f)) todo <- todo[todo >= f]
    message("PUC ", sid, ": ", length(todo), " sampled days to fetch")
    for (d in as.list(todo)) {
      r <- with_retry(function() fetch_env_day(sid, d))
      done <- rbindlist(list(done, data.table(station_id = as.integer(sid), date = d, version = PUC_CACHE_VERSION)))
      if (!nrow(r)) next
      if (d == todo[length(todo)]) {
        message(sprintf("  %s: %d readings, %s to %s UTC", d, nrow(r), min(r$ts), max(r$ts)))
      }
      r <- r[!is.na(temperature) & temperature >= PUC_TEMP_RANGE[1] & temperature <= PUC_TEMP_RANGE[2]]
      if (!nrow(r)) next
      r[, time := as.POSIXct(trunc(ts, "hours"))]
      out[[length(out) + 1]] <- r[, .(
        station_id = as.integer(sid),
        temp       = mean(temperature),
        humidity   = mean(humidity, na.rm = TRUE),
        n          = .N
      ), by = time]
    }
    # Checkpoint per station so an interrupted run resumes cleanly.
    p <- unique(rbindlist(out, use.names = TRUE), by = c("station_id", "time"), fromLast = TRUE)
    if (nrow(p)) setkey(p, station_id, time)
    write_pq(p, cache_path("puc_hourly.parquet"))
    write_pq(unique(done), cache_path("puc_days_done.parquet"))
    out <- list(p)
  }
  out[[1]]
}

# ---------------------------------------------------------------------------
# ERA5-Land hourly 2 m temperature at each PUC station's coordinates.
# Open-Meteo downscales to the point using a 90 m DEM, so the reference is
# already corrected to station elevation (returned as `elevation`).
# ---------------------------------------------------------------------------
update_era5_hourly <- function(stations, puc_hourly) {
  if (is.null(puc_hourly) || !nrow(puc_hourly)) return(NULL)
  rng <- puc_hourly[, .(from = as.Date(min(time)), to = as.Date(max(time))), by = station_id]
  st  <- stations[rng, on = "station_id"]
  res <- lapply(seq_len(nrow(st)), function(i) {
    url <- sprintf(paste0(
      "https://archive-api.open-meteo.com/v1/archive?latitude=%.5f&longitude=%.5f",
      "&start_date=%s&end_date=%s&hourly=temperature_2m&models=era5_land&timezone=GMT"
    ), st$station_lat[i], st$station_lon[i], st$from[i], st$to[i])
    j <- with_retry(function() jsonlite::fromJSON(url))
    Sys.sleep(0.5)
    data.table(
      station_id = st$station_id[i],
      time       = as.POSIXct(j$hourly$time, format = "%Y-%m-%dT%H:%M", tz = "UTC"),
      era5       = j$hourly$temperature_2m,
      elevation  = j$elevation
    )
  })
  e <- rbindlist(res)[puc_hourly[, .(station_id, time)], on = .(station_id, time), nomatch = NULL]
  setkey(e, station_id, time)
  write_pq(e, cache_path("era5_hourly.parquet"))
  e
}
