# Fetch BirdWeather + weather data and maintain an incremental Parquet cache.
#
# Cache layout (all zstd-compressed Parquet via nanoparquet):
#   stations.parquet           one row per Forsyth station
#   counts/<year>.parquet      station_id, date, species_id, count (all int32/Date)
#   progress.parquet           station_id, fetched_through
#   species.parquet            species_id, common_name, scientific_name
#   weather_era5.parquet       date, tmin, tmean, tmax (deg C), rhmin, rhmean, rhmax (%), Open-Meteo ERA5
#   puc_hourly.parquet         station_id, time (UTC hour), temp, humidity, n
#   puc_days_done.parquet      station_id, date (sampled local days fetched), version
#   era5_hourly.parquet        station_id, time, era5, era5_td (deg C), era5_rh (%), elevation; ERA5-Land
#   cocorahs_stations.parquet  sid, name, lat, lon, elev (ft); CoCoRaHS gauges in the county (RCC-ACIS)
#   cocorahs_daily.parquet     sid, date (observation morning), pcpn (in), trace, flag
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
    "&daily=temperature_2m_min,temperature_2m_mean,temperature_2m_max",
    ",relative_humidity_2m_min,relative_humidity_2m_mean,relative_humidity_2m_max"
  ), ctr[["lat"]], ctr[["lon"]], from, to, utils::URLencode(TZ_LOCAL, reserved = TRUE))
  j <- with_retry(function() jsonlite::fromJSON(url))
  w <- data.table(
    date   = as.Date(j$daily$time),
    tmin   = j$daily$temperature_2m_min,
    tmean  = j$daily$temperature_2m_mean,
    tmax   = j$daily$temperature_2m_max,
    rhmin  = j$daily$relative_humidity_2m_min,
    rhmean = j$daily$relative_humidity_2m_mean,
    rhmax  = j$daily$relative_humidity_2m_max
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
# local days, each pulled in full (see fetch_env_day) so diurnal cycles are
# complete. Raw readings
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
# birdweatheR::get_environment_data() post-filters to the UTC day, so this
# calls the same GraphQL query directly and filters to the local date here.
fetch_env_day <- function(sid, d) {
  # ghql rejects declared variables that are not supplied, so the first page
  # uses a query without $after (as birdweatheR does).
  q <- function(after) sprintf('
    query station($id: ID!, $period: InputDuration, $first: Int%s) {
      station(id: $id) {
        sensors {
          environmentHistory(period: $period, first: $first%s) {
            pageInfo { hasNextPage endCursor }
            edges { node { timestamp temperature humidity } }
          }
        }
      }
    }', if (after) ", $after: String" else "", if (after) ", after: $after" else "")
  # birdweatheR:::fetch_page_with_retry() executes query_exec$url_link.
  first_q <- ghql::Query$new()$query("url_link", q(FALSE))
  next_q  <- ghql::Query$new()$query("url_link", q(TRUE))
  # Observed API behavior: the period starts at local midnight of the `from`
  # date but ends at the `to` instant, so `to` must be the local end of day
  # in UTC (03:59:59Z or 04:59:59Z the next day, depending on DST).
  end_local <- as.POSIXct(paste(d + 1L, "00:00:00"), tz = TZ_LOCAL) - 1
  vars <- list(id = as.character(sid), first = 250L,
               period = list(from = paste0(d, "T00:00:00.000Z"),
                             to   = format(end_local, "%Y-%m-%dT%H:%M:%S.000Z", tz = "UTC")))
  pages <- list()
  repeat {
    res <- birdweatheR:::fetch_page_with_retry(if (is.null(vars$after)) first_q else next_q, vars)
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
PUC_CACHE_VERSION <- "local-day-v3"

update_puc_hourly <- function(stations, counts = NULL, days = puc_sample_days()) {
  puc  <- stations[station_type == "puc", station_id]
  old  <- read_pq(cache_path("puc_hourly.parquet"))
  done <- read_pq(cache_path("puc_days_done.parquet"))
  # Earlier versions stored days missing local 20:00-23:59 (v1, v2).
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
# ERA5-Land hourly 2 m temperature, relative humidity, and dew point at each
# PUC station's coordinates.
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
      "&start_date=%s&end_date=%s&models=era5_land&timezone=GMT",
      "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m"
    ), st$station_lat[i], st$station_lon[i], st$from[i], st$to[i])
    j <- with_retry(function() jsonlite::fromJSON(url))
    Sys.sleep(0.5)
    data.table(
      station_id = st$station_id[i],
      time       = as.POSIXct(j$hourly$time, format = "%Y-%m-%dT%H:%M", tz = "UTC"),
      era5       = j$hourly$temperature_2m,
      era5_rh    = j$hourly$relative_humidity_2m,
      era5_td    = j$hourly$dew_point_2m,
      elevation  = j$elevation
    )
  })
  e <- rbindlist(res)[puc_hourly[, .(station_id, time)], on = .(station_id, time), nomatch = NULL]
  setkey(e, station_id, time)
  write_pq(e, cache_path("era5_hourly.parquet"))
  e
}

# ---------------------------------------------------------------------------
# CoCoRaHS daily precipitation for every gauge in Forsyth County (FIPS 37067)
# from the RCC-ACIS web services (no key). One MultiStnData call returns all
# county stations; the full history is refetched each run because observers
# often submit or correct reports days late.
#
# ACIS dates a CoCoRaHS report by its observation morning (about 07:00 local),
# so the value for date d is the 24 h total ending on the morning of d.
# Value codes: "T" trace, "M" missing, "S" included in a later accumulated
# report, and an "A" suffix marks a multi-day accumulated total. Only single-day
# reports become numbers; S and A rows keep pcpn = NA with the code in `flag`.
# ---------------------------------------------------------------------------
ACIS_COUNTY <- "37067"
ACIS_COCORAHS <- "10"  # ACIS station-id type code for the CoCoRaHS network

parse_acis_pcpn <- function(v) {
  v <- trimws(as.character(v))
  flag <- fifelse(v %chin% c("M", "S", "T"), v, fifelse(grepl("A$", v), "A", ""))
  pcpn <- suppressWarnings(as.numeric(v))
  pcpn[v == "T"] <- 0
  list(pcpn = pcpn, trace = v == "T", flag = flag)
}

update_cocorahs <- function(from = HISTORY_START, to = Sys.Date()) {
  url <- sprintf(paste0(
    "https://data.rcc-acis.org/MultiStnData?county=%s&sdate=%s&edate=%s",
    "&elems=pcpn&meta=name,sids,ll,elev&output=json"
  ), ACIS_COUNTY, from, to)
  j <- with_retry(function() jsonlite::fromJSON(url, simplifyVector = FALSE))
  if (!is.null(j$error)) stop("ACIS: ", j$error)
  days <- seq(as.Date(from), as.Date(to), by = "day")
  st <- list()
  out <- list()
  for (s in j$data) {
    sids <- unlist(s$meta$sids)
    ids <- sub(" .*$", "", sids)
    cid <- ids[sub("^.* ", "", sids) == ACIS_COCORAHS | grepl("^NC[A-Z]{2}[0-9]{4}$", ids)]
    if (!length(cid)) next  # COOP, ASOS, and other networks
    v <- vapply(s$data, function(x) as.character(x[[1]]), character(1))
    p <- parse_acis_pcpn(v)
    d <- data.table(sid = cid[1], date = days[seq_along(v)],
                    pcpn = p$pcpn, trace = p$trace, flag = p$flag)[flag != "M"]
    out[[length(out) + 1]] <- d
    ll <- unlist(s$meta$ll)
    st[[length(st) + 1]] <- data.table(
      sid = cid[1], name = s$meta$name,
      lon = if (length(ll) == 2) ll[1] else NA_real_,
      lat = if (length(ll) == 2) ll[2] else NA_real_,
      elev = if (is.null(s$meta$elev)) NA_real_ else as.numeric(s$meta$elev)
    )
  }
  st <- rbindlist(st)
  d <- rbindlist(out)
  if (!nrow(d)) {
    message("No CoCoRaHS reports returned by ACIS")
    return(invisible(NULL))
  }
  setkey(st, sid)
  setkey(d, sid, date)
  write_pq(st, cache_path("cocorahs_stations.parquet"))
  write_pq(d, cache_path("cocorahs_daily.parquet"))
  message(sprintf("CoCoRaHS: %d gauges, %s station-day reports",
                  nrow(st), format(nrow(d), big.mark = ",")))
  d
}
