"""Phase 0 calibration honesty simulation.

Question: when the app says a calibration is Good, is the GPS dot really within ~25 m?

For every corpus map with exact georeferencing, simulate users placing control points:
- tap error: Gaussian in pixels (finger on a phone screen, mapped back to the image)
- position error: Gaussian in metres (0 for printed grid coordinates; ~5 m for an
  "I'm here" GPS fix in good conditions; ~15 m under tree cover)
- layouts: well spread vs clustered

Then compare the calibration's *reported* status/error with its *true* error, measured
on a dense grid of check points over the whole map.

Usage: python experiments/calib_sim.py [runs_per_case]
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict, namedtuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trek_lab.calibration import ControlPoint, Status, calibrate  # noqa: E402
from trek_lab.groundtruth import load_all  # noqa: E402

CORPUS = os.path.join(os.path.dirname(__file__), "..", "corpus")
R_EARTH = 6371008.8

Row = namedtuple("Row", "map_id scenario status_loo status_resid loo resid resid_up true_rms true_max")

SCENARIOS = [
    # name, n_points, layout, tap_sigma_px, pos_sigma_m
    ("grid 4 spread", 4, "spread", 4, 0),
    ("grid 6 spread", 6, "spread", 4, 0),
    ("im_here 2 spread", 2, "spread", 8, 5),
    ("im_here 3 spread", 3, "spread", 8, 5),
    ("im_here 4 spread", 4, "spread", 8, 5),
    ("im_here 6 spread", 6, "spread", 8, 5),
    ("im_here 6 forest", 6, "spread", 8, 15),
    ("im_here 4 clustered", 4, "cluster", 8, 5),
    ("im_here 6 clustered", 6, "cluster", 8, 5),
    ("im_here 8 bad tap", 8, "spread", 40, 5),
]


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def pick_points(rng, n, layout, w, h):
    if layout == "spread":
        # stratified: one point per cell of a coarse grid, inside a 5% margin
        cols = math.ceil(math.sqrt(n * w / h))
        rows = math.ceil(n / cols)
        cells = rng.choice(rows * cols, n, replace=False)
        pts = []
        for c in cells:
            r, k = divmod(c, cols)
            x = (k + rng.uniform(0.15, 0.85)) / cols
            y = (r + rng.uniform(0.15, 0.85)) / rows
            pts.append((0.05 * w + 0.9 * w * x, 0.05 * h + 0.9 * h * y))
        return np.array(pts)
    # cluster: all points inside a 25% x 25% window
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    return np.column_stack([cx + rng.uniform(-0.125, 0.125, n) * w,
                            cy + rng.uniform(-0.125, 0.125, n) * h])


def offset_latlon(lat, lon, de, dn):
    dlat = np.degrees(dn / R_EARTH)
    dlon = np.degrees(de / (R_EARTH * np.cos(np.radians(lat))))
    return lat + dlat, lon + dlon


def run(runs: int = 200, seed: int = 7):
    maps = load_all(CORPUS)
    rng = np.random.default_rng(seed)
    rows = []
    for map_id, (truth, (w, h)) in maps.items():
        gx, gy = np.meshgrid(np.linspace(0.02 * w, 0.98 * w, 25), np.linspace(0.02 * h, 0.98 * h, 25))
        gx, gy = gx.ravel(), gy.ravel()
        glat, glon = truth(gx, gy)
        for name, n, layout, tap_px, pos_m in SCENARIOS:
            for _ in range(runs):
                true_px = pick_points(rng, n, layout, w, h)
                lat, lon = truth(true_px[:, 0], true_px[:, 1])
                lat, lon = offset_latlon(lat, lon, rng.normal(0, pos_m, n), rng.normal(0, pos_m, n))
                tapped = true_px + rng.normal(0, tap_px, true_px.shape)
                cps = [ControlPoint((float(x), float(y)), float(a), float(b))
                       for (x, y), a, b in zip(tapped, lat, lon)]
                cal = calibrate(cps, (w, h), method="loo")
                status_resid = calibrate(cps, (w, h), method="residual").status
                if cal.transform is None:
                    rows.append(Row(map_id, name, cal.status, status_resid, None, None, None, math.nan, math.nan))
                    continue
                en = cal.transform.apply(np.column_stack([gx, gy]))
                plat, plon = cal.plane.latlon_array(en[:, 0], en[:, 1])
                err = haversine(glat, glon, plat, plon)
                rows.append(Row(map_id, name, cal.status, status_resid, cal.loo_rmse_m,
                                cal.resid_rmse_m, cal.resid_upper_m,
                                float(np.sqrt(np.mean(err ** 2))), float(np.max(err))))
    return maps, rows


def _pct(xs, pred):
    return 100 * sum(1 for x in xs if pred(x)) / len(xs) if xs else float("nan")


def report(maps, rows):
    print(f"maps: {', '.join(maps)}\n")
    by = defaultdict(list)
    for r in rows:
        by[r.scenario].append(r)

    print("A. True map-wide RMS error of the dot (metres)")
    print(f"{'scenario':22} {'median':>8} {'p90':>8} {'max-err p90':>12}")
    for name, *_ in SCENARIOS:
        rms = [r.true_rms for r in by[name] if not math.isnan(r.true_rms)]
        mx = [r.true_max for r in by[name] if not math.isnan(r.true_max)]
        print(f"{name:22} {np.median(rms):8.1f} {np.percentile(rms, 90):8.1f} {np.percentile(mx, 90):12.1f}")

    print("\nB. Status honesty: % rated Good, and how many 'Good' were really worse than 25 m")
    print(f"{'scenario':22} {'LOO good%':>10} {'LOO bad-good':>13} {'RESID good%':>12} {'RESID bad-good':>15}")
    for name, *_ in SCENARIOS:
        rs = by[name]
        cols = []
        for attr in ("status_loo", "status_resid"):
            good = [r for r in rs if getattr(r, attr) is Status.GOOD]
            bad = sum(r.true_rms > 25 for r in good)
            cols += [_pct(rs, lambda r, a=attr: getattr(r, a) is Status.GOOD), f"{bad}/{len(good)}"]
        print(f"{name:22} {cols[0]:10.0f} {cols[1]:>13} {cols[2]:12.0f} {cols[3]:>15}")

    print("\nC. Estimator calibration (runs with 4+ points): estimate / true error")
    print(f"{'scenario':22} {'LOO med':>8} {'LOO <true%':>11} {'RES med':>8} {'RES <true%':>11} {'UP med':>7} {'UP <true%':>10}")
    for name, *_ in SCENARIOS:
        rs = [r for r in by[name] if r.loo is not None and r.resid is not None]
        if not rs:
            continue
        out = []
        for attr in ("loo", "resid", "resid_up"):
            ratio = [getattr(r, attr) / r.true_rms for r in rs]
            out += [np.median(ratio), _pct(ratio, lambda x: x < 1)]
        print(f"{name:22} {out[0]:8.2f} {out[1]:11.0f} {out[2]:8.2f} {out[3]:11.0f} {out[4]:7.2f} {out[5]:10.0f}")

    print("\nD. Per map: median true RMS error over all scenarios (m)")
    for map_id in maps:
        rms = [r.true_rms for r in rows if r.map_id == map_id and not math.isnan(r.true_rms)]
        print(f"  {map_id:32} {np.median(rms):6.1f}")


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    report(*run(runs))
