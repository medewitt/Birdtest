# Project constants shared by the cache pipeline and the dashboard.

SEED <- 1834L
set.seed(SEED)

CACHE_DIR <- Sys.getenv("BW_CACHE_DIR", "data/cache")
HISTORY_START <- as.Date("2021-01-01")  # earlier than any BirdWeather station
TZ_LOCAL <- "America/New_York"

# Forsyth County, NC boundary from maps::county (39 vertices, ~1 km precision).
# The bbox is used for the API query; the polygon refines station membership.
forsyth_polygon <- function() {
  m <- maps::map("county", "north carolina,forsyth", plot = FALSE)
  ok <- !is.na(m$x)
  data.table::data.table(lon = m$x[ok], lat = m$y[ok])
}

forsyth_bbox <- function(pad = 0.01) {
  p <- forsyth_polygon()
  list(
    ne = list(lat = max(p$lat) + pad, lon = max(p$lon) + pad),
    sw = list(lat = min(p$lat) - pad, lon = min(p$lon) - pad)
  )
}

# County centroid for gridded weather (polygon vertex mean is close enough
# for a 0.1 deg ERA5 cell).
forsyth_centroid <- function() {
  p <- forsyth_polygon()
  c(lat = mean(p$lat), lon = mean(p$lon))
}

# Ray-casting point-in-polygon; avoids an sf dependency.
in_polygon <- function(x, y, px, py) {
  n <- length(px)
  inside <- logical(length(x))
  j <- n
  for (i in seq_len(n)) {
    hit <- ((py[i] > y) != (py[j] > y)) &
      (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i])
    inside <- xor(inside, hit)
    j <- i
  }
  inside
}
