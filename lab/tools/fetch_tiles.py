"""Stitch a small block of Web-Mercator map tiles into one image, with exact georeferencing.

Usage: python tools/fetch_tiles.py <id> <provider> <lat> <lon> <zoom> <n_tiles> corpus/images corpus/tiles_meta.json

The saved metadata stores the zoom and the top-left tile, so any pixel converts exactly to
lat/lon — free ground truth for calibration tests. Fetches are kept small and slow to
respect each provider's usage policy.
"""

import json
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

UA = {"User-Agent": "TrekFinderCorpusBuilder/0.1 (research; github yuvidewan/TrekFinder)"}

PROVIDERS = {
    "swisstopo": {
        "url": "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg",
        "license": "swisstopo OGD (free use with attribution)", "attribution": "© swisstopo",
    },
    "gsi_japan": {
        "url": "https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
        "license": "GSI Japan tiles (free use with attribution)", "attribution": "出典：国土地理院",
    },
    "opentopomap": {
        "url": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        "license": "CC-BY-SA (OpenTopoMap) / ODbL (OpenStreetMap data)",
        "attribution": "© OpenTopoMap (CC-BY-SA), © OpenStreetMap contributors",
    },
}

TILE = 256


def latlon_to_tile(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def pixel_to_latlon(px: float, py: float, z: int, tx0: int, ty0: int) -> tuple[float, float]:
    """Pixel in the stitched image -> WGS84, given the top-left tile indices."""
    n = 2 ** z
    x = tx0 + px / TILE
    y = ty0 + py / TILE
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def main(map_id, provider, lat, lon, z, n_tiles, out_dir, meta_path):
    lat, lon, z, k = float(lat), float(lon), int(z), int(n_tiles)
    prov = PROVIDERS[provider]
    cx, cy = latlon_to_tile(lat, lon, z)
    tx0, ty0 = int(cx) - k // 2, int(cy) - k // 2
    canvas = np.zeros((k * TILE, k * TILE, 3), np.uint8)
    for j in range(k):
        for i in range(k):
            url = prov["url"].format(z=z, x=tx0 + i, y=ty0 + j)
            data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
            tile = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            canvas[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE] = tile
            time.sleep(0.3)
    fn = f"{map_id}.png"
    cv2.imwrite(os.path.join(out_dir, fn), canvas)
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else []
    meta = [m for m in meta if m["id"] != map_id]
    meta.append({"id": map_id, "file": "images/" + fn, "category": "tile_topo",
                 "provider": provider, "license": prov["license"], "attribution": prov["attribution"],
                 "georef": {"type": "web_mercator_tiles", "zoom": z, "tile_x0": tx0, "tile_y0": ty0,
                            "tile_size": TILE},
                 "center": [lat, lon]})
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(fn, canvas.shape)


if __name__ == "__main__":
    main(*sys.argv[1:9])
