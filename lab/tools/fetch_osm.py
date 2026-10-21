"""Fetch OpenStreetMap trails, roads and waterways covering each georeferenced corpus map.

Used as approximate ground truth for line extraction. OSM is current, while some
corpus maps are decades old, so differences are expected; scores are comparative.
Usage: python tools/fetch_osm.py corpus            (writes corpus/osm/<map_id>.json)
Data © OpenStreetMap contributors, ODbL.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trek_lab.groundtruth import load_all  # noqa: E402

UA = {"User-Agent": "TrekFinderCorpusBuilder/0.1 (research; github yuvidewan/TrekFinder)"}
URL = "https://overpass-api.de/api/interpreter"

TRAIL = {"path", "footway", "track", "bridleway", "steps", "via_ferrata"}
WATER = {"river", "stream", "canal"}


def classify(tags):
    if tags.get("waterway") in WATER:
        return "water"
    hw = tags.get("highway")
    if hw in TRAIL:
        return "trail"
    if hw:
        return "road"
    return None


def main(corpus):
    out_dir = os.path.join(corpus, "osm")
    os.makedirs(out_dir, exist_ok=True)
    for map_id, (truth, (w, h)) in load_all(corpus).items():
        path = os.path.join(out_dir, f"{map_id}.json")
        if os.path.exists(path):
            continue
        xs, ys = np.meshgrid(np.linspace(0, w, 9), np.linspace(0, h, 9))
        lat, lon = truth(xs.ravel(), ys.ravel())
        s, n, wl, e = lat.min(), lat.max(), lon.min(), lon.max()
        q = (f"[out:json][timeout:120];("
             f'way["highway"]({s},{wl},{n},{e});'
             f'way["waterway"~"^(river|stream|canal)$"]({s},{wl},{n},{e}););out geom;')
        req = urllib.request.Request(URL, data=urllib.parse.urlencode({"data": q}).encode(), headers=UA)
        data = json.load(urllib.request.urlopen(req, timeout=180))
        feats = []
        for el in data["elements"]:
            cls = classify(el.get("tags", {}))
            if cls and "geometry" in el:
                feats.append({"class": cls, "name": el["tags"].get("name"),
                              "latlon": [[g["lat"], g["lon"]] for g in el["geometry"]]})
        json.dump({"bbox": [s, wl, n, e], "source": "OpenStreetMap contributors (ODbL)",
                   "features": feats}, open(path, "w", encoding="utf-8"))
        counts = {c: sum(f["class"] == c for f in feats) for c in ("trail", "road", "water")}
        print(map_id, counts)
        time.sleep(5)


if __name__ == "__main__":
    main(sys.argv[1])
