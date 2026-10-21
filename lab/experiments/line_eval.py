"""Score trail and water extraction against OpenStreetMap on georeferenced corpus maps.

For each map and class (trail, water):
- the ink layer is chosen as a user would by tapping the feature once: we take the
  layer that covers the most ground-truth pixels ("one-tap oracle");
- centrelines are extracted from that layer (trek_lab.lines);
- recall    = share of OSM line length within TOL metres of an extracted line;
- precision = share of extracted line length within TOL metres of ANY OSM line
  (trail/road/water), because one ink colour often draws several feature types.
Baselines: the same scores using ALL ink (no layer choice).

Caveat: OSM is 2020s data; USGS sheets are from 1971/1997, so trails have moved,
appeared or vanished. Treat absolute numbers on old maps as lower bounds.
Usage: python experiments/line_eval.py [max_edge_px]
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trek_lab.groundtruth import load_all, load_inverse_all  # noqa: E402
from trek_lab.imageprep import extract_ink_layers, load  # noqa: E402
from trek_lab.lines import centrelines  # noqa: E402

LAB = os.path.join(os.path.dirname(__file__), "..")
CORPUS = os.path.join(LAB, "corpus")
OUT = os.path.join(LAB, "out", "lines")
TOL_M = 25.0


def rasterise(feats, inv, scale, shape, cls):
    m = np.zeros(shape, np.uint8)
    for f in feats:
        if cls != "any" and f["class"] != cls:
            continue
        ll = np.array(f["latlon"])
        x, y = inv(ll[:, 0], ll[:, 1])
        pts = np.round(np.column_stack([x, y]) * scale).astype(np.int32)
        cv2.polylines(m, [pts], False, 255, 1)
    return m > 0


def near(a, b, tol_px):
    """Share of True pixels in a that lie within tol_px of a True pixel in b."""
    if a.sum() == 0:
        return float("nan")
    dist = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dist[a] <= tol_px).mean())


def main(max_edge=4000):
    os.makedirs(OUT, exist_ok=True)
    manifest = {m["id"]: m for m in json.load(open(os.path.join(CORPUS, "manifest.json"), encoding="utf-8"))["maps"]}
    fwd, inv_all = load_all(CORPUS), load_inverse_all(CORPUS)
    print(f"{'map':30} {'class':6} {'layer':>5} {'m/px':>5} | {'recall':>6} {'prec':>5} | {'all-ink R':>9} {'all-ink P':>9}")
    for map_id, inv in inv_all.items():
        m = manifest[map_id]
        _, (gw, gh) = fwd[map_id]
        img = load(os.path.join(CORPUS, m["file"]), m.get("page"), max_edge=max_edge)
        scale = img.shape[1] / gw
        feats = json.load(open(os.path.join(CORPUS, "osm", f"{map_id}.json"), encoding="utf-8"))["features"]
        # metres per pixel at this scale, from the ground truth
        la, lo = fwd[map_id][0](np.array([gw / 2, gw / 2 + 100.0]), np.array([gh / 2, gh / 2]))
        mpp = float(np.hypot((la[1] - la[0]) * 111320, (lo[1] - lo[0]) * 111320 * np.cos(np.radians(la[0]))) / 100 / scale)
        tol = TOL_M / mpp
        layers = extract_ink_layers(img)
        any_gt = rasterise(feats, inv, scale, img.shape[:2], "any")
        all_ink_lines = centrelines(layers.ink_mask)
        overlay = cv2.addWeighted(img, 0.45, np.full_like(img, 255), 0.55, 0)
        for cls, colour in (("trail", (0, 160, 0)), ("water", (255, 120, 0))):
            gt = rasterise(feats, inv, scale, img.shape[:2], cls)
            if gt.sum() < 200:
                continue
            # One-tap oracle: a user taps the trail a few times; each tap lands on the ink
            # pixel nearest a ground-truth point. The most-hit layer is "the trail colour".
            labels = layers.palette.labels
            rng = np.random.default_rng(0)
            gy, gx = np.nonzero(gt)
            ink_y, ink_x = np.nonzero(labels >= 0)
            votes = []
            for i in rng.choice(len(gx), min(15, len(gx)), replace=False):
                d2 = (ink_x - gx[i]) ** 2 + (ink_y - gy[i]) ** 2
                j = int(np.argmin(d2))
                if d2[j] <= 9:  # within 3 px: the tap actually hit drawn ink
                    votes.append(labels[ink_y[j], ink_x[j]])
            if not votes:
                print(f"{map_id:30} {cls:6}  no ink under any tap")
                continue
            best = int(np.bincount(votes).argmax())
            per_layer = []
            for i in range(len(layers.palette.colors_bgr)):
                li = centrelines(labels == i)
                ri, pi = near(gt, li, tol), near(li, any_gt, tol)
                f1 = 2 * ri * pi / (ri + pi) if ri + pi > 0 else 0.0
                per_layer.append((f1, ri, pi, i))
            f1b, rb, pb, ib = max(per_layer)
            print(f"{'':30} {'':6} best layer {ib}: recall {rb:.0%} prec {pb:.0%}   "
                  f"(layers R/P: {' '.join(f'{i}:{r:.0%}/{p:.0%}' for _, r, p, i in per_layer)})")
            lines = centrelines(labels == best)
            r, p = near(gt, lines, tol), near(lines, any_gt, tol)
            r0, p0 = near(gt, all_ink_lines, tol), near(all_ink_lines, any_gt, tol)
            print(f"{map_id:30} {cls:6} {best:5} {mpp:5.1f} | {r:6.0%} {p:5.0%} | {r0:9.0%} {p0:9.0%}")
            overlay[gt] = colour
            overlay[lines] = (200, 0, 200) if cls == "trail" else (0, 0, 220)
        cv2.imwrite(os.path.join(OUT, f"{map_id}_lines.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
