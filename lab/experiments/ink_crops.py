"""Zoomed check of ink layers: a crop of the source next to each ink layer of that crop.
Thumbnails hide thin lines, so this works at near-native resolution.
Usage: python experiments/ink_crops.py <map_id> <cx_frac> <cy_frac> [size_px]
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trek_lab.imageprep import extract_ink_layers, load  # noqa: E402

LAB = os.path.join(os.path.dirname(__file__), "..")


def main(map_id, cx, cy, size=700):
    m = next(x for x in json.load(open(os.path.join(LAB, "corpus", "manifest.json"), encoding="utf-8"))["maps"]
             if x["id"] == map_id)
    img = load(os.path.join(LAB, "corpus", m["file"]), m.get("page"))
    h, w = img.shape[:2]
    size = int(size)
    x0 = int(np.clip(float(cx) * w - size / 2, 0, w - size))
    y0 = int(np.clip(float(cy) * h - size / 2, 0, h - size))
    crop = img[y0:y0 + size, x0:x0 + size]
    layers = extract_ink_layers(crop)
    pal = layers.palette
    tiles = [crop]
    for i in range(len(pal.colors_bgr)):
        t = np.full(crop.shape, 255, np.uint8)
        t[pal.labels == i] = pal.colors_bgr[i]
        cv2.rectangle(t, (0, 0), (50, 26), pal.colors_bgr[i].tolist(), -1)
        cv2.putText(t, f"{pal.share[i]:.0%} of ink", (56, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.full(crop.shape, 255, np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    os.makedirs(os.path.join(LAB, "out", "prep"), exist_ok=True)
    out = os.path.join(LAB, "out", "prep", f"{map_id}_inkcrop.jpg")
    cv2.imwrite(out, sheet)
    print(out, f"ink {layers.ink_mask.mean():.1%} of crop")


if __name__ == "__main__":
    main(*sys.argv[1:])
