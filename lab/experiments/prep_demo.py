"""Visual check of image preparation on the corpus.

1. Photos: automatic map-outline detection + straightening -> out/prep/<id>_rectify.jpg
2. All maps: 8-colour palette + per-colour layer masks -> out/prep/<id>_palette.jpg
Usage: python experiments/prep_demo.py [map_id ...]
"""

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trek_lab.imageprep import extract_ink_layers, find_map_quad, load, rectify  # noqa: E402

LAB = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(LAB, "out", "prep")


def fit(img, edge):
    s = edge / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def rectify_sheet(map_id, img):
    q = find_map_quad(img)
    vis = img.copy()
    if q.quad is None:
        cv2.putText(vis, "NO MAP OUTLINE FOUND", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
        cv2.imwrite(os.path.join(OUT, f"{map_id}_rectify.jpg"), fit(vis, 1200))
        return "not found"
    cv2.polylines(vis, [q.quad.astype(np.int32)], True, (0, 0, 255), max(4, img.shape[1] // 300))
    flat, _ = rectify(img, q.quad, max_edge=2000)
    a, b = fit(vis, 900), fit(flat, 900)
    canvas = np.full((900, 1800, 3), 255, np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], 900:900 + b.shape[1]] = b
    cv2.imwrite(os.path.join(OUT, f"{map_id}_rectify.jpg"), canvas)
    return f"quad {q.area_frac:.0%} of photo ({q.method})"


def palette_sheet(map_id, img, k=8):
    """Tile 1: source. Tiles 2-7: ink layers (lines/text/symbols by colour).
    Tiles 8-9: paper palette swatches (the map's background style)."""
    small = fit(img, 2400)
    t = time.time()
    layers = extract_ink_layers(small, k_ink=6, k_paper=4)
    dt = time.time() - t
    tiles = [fit(small, 400)]
    pal = layers.palette
    for i in range(6):
        mask = np.full(small.shape, 255, np.uint8)
        mask[pal.labels == i] = pal.colors_bgr[i]
        m = fit(mask, 400)
        cv2.rectangle(m, (0, 0), (60, 30), pal.colors_bgr[i].tolist(), -1)
        cv2.putText(m, f"ink {pal.share[i] * layers.ink_mask.mean():.1%}", (66, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        tiles.append(m)
    for half in (layers.paper.colors_bgr[:2], layers.paper.colors_bgr[2:]):
        sw = np.full((400, 400, 3), 255, np.uint8)
        for j, c in enumerate(half):
            sw[j * 200:(j + 1) * 200] = c
        cv2.putText(sw, "paper", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        tiles.append(sw)
    hmax = max(t_.shape[0] for t_ in tiles)
    tiles = [np.vstack([t_, np.full((hmax - t_.shape[0], t_.shape[1], 3), 255, np.uint8)]) for t_ in tiles]
    wmax = max(t_.shape[1] for t_ in tiles)
    tiles = [np.hstack([t_, np.full((hmax, wmax - t_.shape[1], 3), 255, np.uint8)]) for t_ in tiles]
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])
    cv2.imwrite(os.path.join(OUT, f"{map_id}_palette.jpg"), grid)
    return dt


def main(ids):
    os.makedirs(OUT, exist_ok=True)
    maps = json.load(open(os.path.join(LAB, "corpus", "manifest.json"), encoding="utf-8"))["maps"]
    for m in maps:
        if ids and m["id"] not in ids:
            continue
        t = time.time()
        img = load(os.path.join(LAB, "corpus", m["file"]), m.get("page"))
        load_s = time.time() - t
        msg = rectify_sheet(m["id"], img) if m["category"] == "photo" else ""
        pal_s = palette_sheet(m["id"], img)
        print(f"{m['id']:36} {img.shape[1]:5}x{img.shape[0]:<5} load {load_s:4.1f}s palette {pal_s:4.1f}s {msg}")


if __name__ == "__main__":
    main(sys.argv[1:])
