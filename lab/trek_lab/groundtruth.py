"""Exact pixel -> WGS84 ground truth for corpus maps that carry georeferencing.

Supported:
- GeoTIFF (USGS topo sheets): affine geotransform + projected CRS, reprojected to WGS84.
- Web-Mercator tile mosaics written by tools/fetch_tiles.py.
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable

import numpy as np

LatLonFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def from_geotiff(path: str) -> tuple[LatLonFn, tuple[int, int]]:
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as r:
        t, crs, size = r.transform, r.crs, (r.width, r.height)
    to_wgs84 = Transformer.from_crs(crs.to_wkt(), "EPSG:4326", always_xy=True)

    def fn(px, py):
        px, py = np.asarray(px, float), np.asarray(py, float)
        # +0.5: rasterio's transform maps pixel corners; we use pixel centres.
        x = t.c + (px + 0.5) * t.a + (py + 0.5) * t.b
        y = t.f + (px + 0.5) * t.d + (py + 0.5) * t.e
        lon, lat = to_wgs84.transform(x, y)
        return np.asarray(lat), np.asarray(lon)

    return fn, size


def from_tiles(georef: dict, size: tuple[int, int]) -> tuple[LatLonFn, tuple[int, int]]:
    z, tx0, ty0, ts = georef["zoom"], georef["tile_x0"], georef["tile_y0"], georef["tile_size"]
    n = 2 ** z

    def fn(px, py):
        x = tx0 + (np.asarray(px, float) + 0.5) / ts
        y = ty0 + (np.asarray(py, float) + 0.5) / ts
        lon = x / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * y / n))))
        return lat, lon

    return fn, size


def load_all(corpus_dir: str) -> dict[str, tuple[LatLonFn, tuple[int, int]]]:
    """Every corpus map with exact georeferencing, keyed by map id."""
    import cv2

    out = {}
    manifest = json.load(open(os.path.join(corpus_dir, "manifest.json"), encoding="utf-8"))
    for m in manifest["maps"]:
        path = os.path.join(corpus_dir, m["file"])
        if m["file"].endswith(".tif"):
            out[m["id"]] = from_geotiff(path)
        elif m.get("georef", {}).get("type") == "web_mercator_tiles":
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            out[m["id"]] = from_tiles(m["georef"], (img.shape[1], img.shape[0]))
    return out
