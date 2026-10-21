"""Static terrain and drainage layer for Forsyth County, written to data/terrain.parquet.

Source: USGS 3D Elevation Program (3DEP) dynamic elevation service
(elevation.nationalmap.gov, ImageServer exportImage), resampled bilinearly to
a 3 arc-second grid (about 92 m north-south, 75 m east-west) over BBOX.

Derived layers, computed on the 3 arc-second grid:

elev_m     elevation (m above NAVD88)
tpi_m      topographic position index: elevation minus the mean elevation in
           a box of about 2 x 2 km around the cell. Negative values are
           valleys and hollows where cold air drains and pools on clear,
           calm nights; positive values are ridges and knolls.
drain_km2  upstream drainage area from D8 flow routing on the depression-
           filled DEM (priority-flood with a small epsilon so that flats
           drain).
stream     drainage area >= STREAM_KM2. This is a terrain-derived channel
           network, not the mapped NHD network; small headwater channels and
           piped urban streams differ.
hand_m     height above nearest drainage: elevation minus the elevation of
           the first stream cell reached by following D8 flow downhill.
           Cells whose flow leaves the grid before reaching a stream are NA.

The written grid is the 3 x 3 block aggregate (0.0025 deg, about 280 x 225 m):
mean of elev_m, tpi_m and hand_m, maximum of drain_km2, any of stream.
The layer is static, so it is committed rather than rebuilt by CI.

Run (needs network): uv run --group analysis python scripts/terrain.py
"""

import heapq
import io
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BBOX = (-80.56, 35.95, -79.99, 36.30)  # lon0, lat0, lon1, lat1; pads the county
RES = 1 / 1200                         # 3 arc-seconds
AGG = 3                                # output block size in cells
TPI_RADIUS_M = 1000
STREAM_KM2 = 2.0
EPS = 1e-4                             # m, flat-resolving increment
OUT = Path("data/terrain.parquet")

URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/"
       "exportImage?bbox={},{},{},{}&bboxSR=4326&imageSR=4326&size={},{}&format=tiff"
       "&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image")


def fetch_dem():
    nx = round((BBOX[2] - BBOX[0]) / RES)
    ny = round((BBOX[3] - BBOX[1]) / RES)
    with urllib.request.urlopen(URL.format(*BBOX, nx, ny), timeout=120) as r:
        img = Image.open(io.BytesIO(r.read()))
        z = np.asarray(img, dtype=np.float64)
    if z.shape != (ny, nx):
        raise SystemExit(f"DEM shape {z.shape}, expected {(ny, nx)}")
    if (z < -100).any():
        raise SystemExit("DEM has no-data cells inside the box")
    lon = BBOX[0] + (np.arange(nx) + 0.5) * RES
    lat = BBOX[3] - (np.arange(ny) + 0.5) * RES  # row 0 is north
    return z, lon, lat


def cell_size_m(lat):
    dy = RES * 111_320.0
    dx = RES * 111_320.0 * np.cos(np.deg2rad(lat.mean()))
    return dx, dy


def box_mean(z, hx, hy):
    """Mean over a (2 hy + 1) x (2 hx + 1) box, shrinking at the edges."""
    ny, nx = z.shape
    c = np.zeros((ny + 1, nx + 1))
    c[1:, 1:] = z.cumsum(0).cumsum(1)
    i = np.arange(ny)
    j = np.arange(nx)
    i0, i1 = np.clip(i - hy, 0, ny), np.clip(i + hy + 1, 0, ny)
    j0, j1 = np.clip(j - hx, 0, nx), np.clip(j + hx + 1, 0, nx)
    s = (c[i1][:, j1] - c[i0][:, j1] - c[i1][:, j0] + c[i0][:, j0])
    n = (i1 - i0)[:, None] * (j1 - j0)[None, :]
    return s / n


OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def priority_flood(z):
    """Depression filling (Barnes et al. 2014, epsilon variant), flat index arrays."""
    ny, nx = z.shape
    f = z.copy().ravel()
    done = np.zeros(ny * nx, bool)
    heap = []
    for i in range(ny):
        for j in range(nx):
            if i in (0, ny - 1) or j in (0, nx - 1):
                k = i * nx + j
                heapq.heappush(heap, (f[k], k))
                done[k] = True
    while heap:
        e, k = heapq.heappop(heap)
        i, j = divmod(k, nx)
        for di, dj in OFFS:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx:
                m = a * nx + b
                if not done[m]:
                    done[m] = True
                    if f[m] <= e:
                        f[m] = e + EPS
                    heapq.heappush(heap, (f[m], m))
    return f.reshape(ny, nx)


def d8(f, dx, dy):
    """Receiver of each cell (flat index, -1 drains off-grid) by steepest descent."""
    ny, nx = f.shape
    pad = np.pad(f, 1, constant_values=-np.inf)
    best = np.zeros((ny, nx))
    rec = np.full((ny, nx), -1, np.int64)
    ii, jj = np.mgrid[0:ny, 0:nx]
    for di, dj in OFFS:
        dist = np.hypot(di * dy, dj * dx)
        nb = pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        slope = (f - nb) / dist
        a, b = ii + di, jj + dj
        inside = (a >= 0) & (a < ny) & (b >= 0) & (b < nx)
        # Off-grid neighbours (slope inf) win, so edge cells drain outward.
        better = slope > best
        rec = np.where(better, np.where(inside, a * nx + b, -1), rec)
        best = np.where(better, slope, best)
    return rec.ravel()


def main():
    z, lon, lat = fetch_dem()
    ny, nx = z.shape
    dx, dy = cell_size_m(lat)
    print(f"DEM {ny} x {nx} cells, {dx:.0f} x {dy:.0f} m, elevation {z.min():.0f}-{z.max():.0f} m")

    tpi = z - box_mean(z, round(TPI_RADIUS_M / dx), round(TPI_RADIUS_M / dy))

    f = priority_flood(z)
    rec = d8(f, dx, dy)
    order = np.argsort(f.ravel(), kind="stable")  # ascending filled elevation
    acc = np.ones(ny * nx)
    for k in order[::-1]:
        r = rec[k]
        if r >= 0:
            acc[r] += acc[k]
    drain = acc * dx * dy / 1e6
    stream = drain >= STREAM_KM2
    zf = z.ravel()
    base = np.full(ny * nx, np.nan)
    for k in order:  # receivers are lower, so they are resolved first
        if stream[k]:
            base[k] = zf[k]
        elif rec[k] >= 0:
            base[k] = base[rec[k]]
    hand = np.maximum(zf - base, 0.0).reshape(ny, nx)
    drain = drain.reshape(ny, nx)
    stream = stream.reshape(ny, nx)

    my, mx = ny // AGG, nx // AGG

    def blocks(a):
        return a[:my * AGG, :mx * AGG].reshape(my, AGG, mx, AGG)

    out_lon = lon[:mx * AGG].reshape(mx, AGG).mean(1)
    out_lat = lat[:my * AGG].reshape(my, AGG).mean(1)
    gx, gy = np.meshgrid(out_lon, out_lat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # blocks with no stream path
        hand_b = np.nanmean(blocks(hand), axis=(1, 3))
    tab = pa.table({
        "lon": np.round(gx.ravel(), 5),
        "lat": np.round(gy.ravel(), 5),
        "elev_m": np.round(blocks(z).mean(axis=(1, 3)).ravel(), 1).astype(np.float32),
        "tpi_m": np.round(blocks(tpi).mean(axis=(1, 3)).ravel(), 2).astype(np.float32),
        "hand_m": np.round(hand_b.ravel(), 2).astype(np.float32),
        "drain_km2": np.round(blocks(drain).max(axis=(1, 3)).ravel(), 3).astype(np.float32),
        "stream": blocks(stream).any(axis=(1, 3)).ravel(),
    })
    pq.write_table(tab, OUT, compression="zstd")
    print(f"Wrote {OUT}: {tab.num_rows} cells, {int(np.sum(tab['stream']))} stream cells")


if __name__ == "__main__":
    main()
