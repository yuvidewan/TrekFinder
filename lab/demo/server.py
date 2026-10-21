"""TrekFinder engine demo: a local web UI over the Phase 0 lab code.

Run via run_demo.bat (repo root), or: .venv/Scripts/python demo/server.py
Serves on http://127.0.0.1:8765 only (not reachable from other machines).
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, LAB)
sys.path.insert(0, os.path.join(LAB, "experiments"))

from trek_lab.calibration import ControlPoint, Status, calibrate  # noqa: E402
from trek_lab.groundtruth import load_all, load_inverse_all  # noqa: E402
from trek_lab.imageprep import extract_ink_layers, find_map_quad, load, rectify  # noqa: E402
from trek_lab.lines import centrelines  # noqa: E402

CORPUS = os.path.join(LAB, "corpus")
UPLOADS = os.path.join(LAB, "out", "uploads")
ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf", ".webp"}
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # decision 18A.2

_lock = threading.Lock()
_cache: dict = {}
_GT = {"fwd": None, "inv": None}


# ---------------------------------------------------------------- helpers

def ground_truth():
    if _GT["fwd"] is None:
        _GT["fwd"], _GT["inv"] = load_all(CORPUS), load_inverse_all(CORPUS)
    return _GT["fwd"], _GT["inv"]


def manifest():
    maps = json.load(open(os.path.join(CORPUS, "manifest.json"), encoding="utf-8"))["maps"]
    out = [m for m in maps if os.path.exists(os.path.join(CORPUS, m["file"]))]
    for fn in sorted(os.listdir(UPLOADS)):
        out.append({"id": "upload:" + fn, "file": os.path.join(UPLOADS, fn), "category": "your upload",
                    "license": "yours", "notes": "Uploaded in the demo"})
    return out


def entry(map_id):
    m = next((m for m in manifest() if m["id"] == map_id), None)
    if m is None:
        raise KeyError(map_id)
    return m


def path_of(m):
    return m["file"] if os.path.isabs(m["file"]) else os.path.join(CORPUS, m["file"])


def jpeg(img, max_edge=1400, quality=82):
    s = max_edge / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def png(img, max_edge=900):
    s = max_edge / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def working(map_id, max_edge):
    key = ("img", map_id, max_edge)
    with _lock:
        if key not in _cache:
            m = entry(map_id)
            _cache[key] = load(path_of(m), m.get("page"), max_edge=max_edge)
        return _cache[key]


def ink(map_id, max_edge):
    key = ("ink", map_id, max_edge)
    img = working(map_id, max_edge)
    with _lock:
        if key not in _cache:
            _cache[key] = extract_ink_layers(img)
        return _cache[key]


def thick(mask, px):
    return cv2.dilate(mask.astype(np.uint8), np.ones((px, px), np.uint8)) > 0


# ---------------------------------------------------------------- routes

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/maps")
def maps():
    fwd, _ = ground_truth()
    return jsonify([{"id": m["id"], "category": m.get("category", "?"), "notes": m.get("notes", ""),
                     "license": m.get("license", ""), "georef": m["id"] in fwd,
                     "osm": os.path.exists(os.path.join(CORPUS, "osm", f"{m['id']}.json"))}
                    for m in manifest()])


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or os.path.splitext(f.filename)[1].lower() not in ALLOWED:
        return jsonify(error="Use a JPG, PNG, TIFF, WebP or PDF file."), 400
    name = "".join(c for c in os.path.basename(f.filename) if c.isalnum() or c in "._-")[-80:]
    f.save(os.path.join(UPLOADS, name))
    return jsonify(id="upload:" + name)


@app.get("/api/preview/<path:map_id>")
def preview(map_id):
    t = time.time()
    img = working(map_id, int(request.args.get("edge", 3000)))
    return jsonify(image=jpeg(img), width=img.shape[1], height=img.shape[0], seconds=round(time.time() - t, 2))


@app.post("/api/straighten/<path:map_id>")
def straighten(map_id):
    img = working(map_id, int(request.args.get("edge", 3000)))
    t = time.time()
    q = find_map_quad(img)
    if q.quad is None:
        return jsonify(found=False, seconds=round(time.time() - t, 2),
                       message="No clear map outline found. In the app the user would drag the 4 corners.")
    vis = img.copy()
    cv2.polylines(vis, [q.quad.astype(np.int32)], True, (0, 0, 255), max(4, img.shape[1] // 250))
    flat, _ = rectify(img, q.quad, max_edge=2400)
    return jsonify(found=True, seconds=round(time.time() - t, 2), area=round(q.area_frac, 3), method=q.method,
                   outlined=jpeg(vis, 1000), flat=jpeg(flat, 1000))


@app.post("/api/layers/<path:map_id>")
def layers(map_id):
    edge = int(request.args.get("edge", 3000))
    t = time.time()
    lay = ink(map_id, edge)
    img = working(map_id, edge)
    pal = lay.palette
    out = []
    for i, c in enumerate(pal.colors_bgr):
        tile = np.full(img.shape, 255, np.uint8)
        tile[pal.labels == i] = c
        out.append({"index": i, "color": "#%02x%02x%02x" % (int(c[2]), int(c[1]), int(c[0])),
                    "share": round(float(pal.share[i]), 4), "image": png(tile, 700)})
    paper = ["#%02x%02x%02x" % (int(c[2]), int(c[1]), int(c[0])) for c in lay.paper.colors_bgr]
    return jsonify(layers=out, paper=paper, ink_share=round(float(lay.ink_mask.mean()), 4),
                   seconds=round(time.time() - t, 2))


@app.post("/api/lines/<path:map_id>")
def lines(map_id):
    edge = int(request.args.get("edge", 3000))
    layer = int(request.args.get("layer", 0))
    img = working(map_id, edge)
    lay = ink(map_id, edge)
    t = time.time()
    cl = centrelines(lay.palette.labels == layer)
    secs = round(time.time() - t, 2)
    vis = cv2.addWeighted(img, 0.4, np.full_like(img, 255), 0.6, 0)
    w_px = max(2, img.shape[1] // 900)
    score = None
    fwd, inv = ground_truth()
    osm_path = os.path.join(CORPUS, "osm", f"{map_id}.json")
    if map_id in inv and os.path.exists(osm_path):
        from line_eval import TOL_M, near, rasterise
        feats = json.load(open(osm_path, encoding="utf-8"))["features"]
        _, (gw, gh) = fwd[map_id]
        scale = img.shape[1] / gw
        la, lo = fwd[map_id][0](np.array([gw / 2, gw / 2 + 100.0]), np.array([gh / 2, gh / 2]))
        mpp = float(np.hypot((la[1] - la[0]) * 111320, (lo[1] - lo[0]) * 111320 * math.cos(math.radians(la[0]))) / 100 / scale)
        tol = TOL_M / mpp
        any_gt = rasterise(feats, inv[map_id], scale, img.shape[:2], "any")
        score = {"metres_per_px": round(mpp, 2), "tolerance_m": TOL_M,
                 "precision": round(near(cl, any_gt, tol), 3)}
        for cls, col in (("trail", (0, 170, 0)), ("water", (230, 140, 0))):
            gt = rasterise(feats, inv[map_id], scale, img.shape[:2], cls)
            vis[thick(gt, w_px)] = col
            score[f"{cls}_recall"] = round(near(gt, cl, tol), 3) if gt.any() else None
    vis[thick(cl, w_px)] = (200, 0, 200)
    return jsonify(image=jpeg(vis, 1600), seconds=secs, score=score)


@app.post("/api/calibrate/<path:map_id>")
def calibrate_demo(map_id):
    """Simulate a user calibrating with N taps, then walking a real trail with noisy GPS."""
    fwd, inv = ground_truth()
    if map_id not in fwd:
        return jsonify(error="Calibration demo needs a map with known real-world positions "
                             "(the USGS sheets or the Swiss / Japan / Everest tile maps)."), 400
    p = request.get_json(force=True)
    n, layout = int(p.get("points", 6)), p.get("layout", "spread")
    tap_px, pos_m, gps_m = float(p.get("tap_px", 8)), float(p.get("point_gps_m", 5)), float(p.get("walk_gps_m", 8))
    rng = np.random.default_rng(int(p.get("seed", 1)))
    import calib_sim as cs

    truth, (gw, gh) = fwd[map_id]
    true_px = cs.pick_points(rng, n, layout, gw, gh)
    lat, lon = truth(true_px[:, 0], true_px[:, 1])
    lat, lon = cs.offset_latlon(lat, lon, rng.normal(0, pos_m, n), rng.normal(0, pos_m, n))
    tapped = true_px + rng.normal(0, tap_px, true_px.shape)
    cal = calibrate([ControlPoint((float(x), float(y)), float(a), float(b))
                     for (x, y), a, b in zip(tapped, lat, lon)], (gw, gh))

    img = working(map_id, 3000)
    s = img.shape[1] / gw
    vis = cv2.addWeighted(img, 0.55, np.full_like(img, 255), 0.45, 0)
    r = max(6, img.shape[1] // 150)
    for x, y in tapped:
        cv2.drawMarker(vis, (int(x * s), int(y * s)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, r * 2, max(2, r // 3))

    result = {"status": cal.status.value, "points": n, "coverage": round(cal.coverage, 2),
              "error_bound_m": None if cal.rmse_m is None else round(cal.rmse_m, 1)}
    if cal.transform is None:
        result["message"] = "Not enough usable points for any position (need 2+ not in a line)."
        return jsonify(image=jpeg(vis, 1600), **result)

    gx, gy = np.meshgrid(np.linspace(0.02 * gw, 0.98 * gw, 25), np.linspace(0.02 * gh, 0.98 * gh, 25))
    glat, glon = truth(gx.ravel(), gy.ravel())
    en = cal.transform.apply(np.column_stack([gx.ravel(), gy.ravel()]))
    plat, plon = cal.plane.latlon_array(en[:, 0], en[:, 1])
    result["true_error_m"] = round(float(np.sqrt(np.mean(cs.haversine(glat, glon, plat, plon) ** 2))), 1)

    # GPS walk along the longest OSM trail fully inside the map
    osm_path = os.path.join(CORPUS, "osm", f"{map_id}.json")
    walk = None
    if os.path.exists(osm_path):
        best = None
        for f in json.load(open(osm_path, encoding="utf-8"))["features"]:
            if f["class"] != "trail":
                continue
            ll = np.array(f["latlon"])
            x, y = inv[map_id](ll[:, 0], ll[:, 1])
            inside = (x > 0.05 * gw) & (x < 0.95 * gw) & (y > 0.05 * gh) & (y < 0.95 * gh)
            if inside.all() and len(ll) > 5:
                length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
                if best is None or length > best[0]:
                    best = (length, ll, x, y)
        if best:
            _, ll, x, y = best
            cv2.polylines(vis, [np.round(np.column_stack([x, y]) * s).astype(np.int32)], False,
                          (0, 160, 0), max(3, r // 2))
            seg = np.hypot(np.diff(x), np.diff(y))
            dist = np.r_[0, np.cumsum(seg)]
            step = 20 / max(cal.transform.metres_per_pixel(), 1e-6)  # a fix every ~20 m
            t = np.arange(0, dist[-1], step)
            wx, wy = np.interp(t, dist, x), np.interp(t, dist, y)
            wlat, wlon = truth(wx, wy)
            wlat, wlon = cs.offset_latlon(wlat, wlon, rng.normal(0, gps_m, len(t)), rng.normal(0, gps_m, len(t)))
            dots = np.array([cal.latlon_to_px(a, b) for a, b in zip(wlat, wlon)])
            cv2.polylines(vis, [np.round(dots * s).astype(np.int32)], False, (200, 0, 200), max(2, r // 3))
            last = tuple(np.round(dots[-1] * s).astype(int))
            rad = int(cal.uncertainty_px(gps_m) * s)
            overlay = vis.copy()
            cv2.circle(overlay, last, rad, (255, 120, 0), -1)
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
            cv2.circle(vis, last, rad, (255, 120, 0), max(2, r // 3))
            cv2.circle(vis, last, r, (255, 255, 255), -1)
            cv2.circle(vis, last, r, (255, 80, 0), max(2, r // 3))
            off = cs.haversine(*truth(wx, wy), *truth(dots[:, 0], dots[:, 1]))
            walk = {"trail_km": round(float(dist[-1]) * cal.transform.metres_per_pixel() / 1000, 2),
                    "fixes": int(len(t)), "dot_error_median_m": round(float(np.median(off)), 1),
                    "dot_error_p90_m": round(float(np.percentile(off, 90)), 1),
                    "circle_radius_m": round(cal.uncertainty_m(gps_m), 1),
                    "inside_circle": round(float(np.mean(off <= cal.uncertainty_m(gps_m))), 3)}
            # zoom on the walk
            pts = np.vstack([np.column_stack([x, y]), dots]) * s
            x0, y0 = np.maximum(pts.min(0) - 250, 0).astype(int)
            x1, y1 = np.minimum(pts.max(0) + 250, [vis.shape[1], vis.shape[0]]).astype(int)
            result["walk_image"] = jpeg(vis[y0:y1, x0:x1], 1200)
    result["walk"] = walk
    return jsonify(image=jpeg(vis, 1600), **result)


if __name__ == "__main__":
    print("TrekFinder engine demo on http://127.0.0.1:8765  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)
