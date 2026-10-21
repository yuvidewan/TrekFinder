import math

import numpy as np
import pytest

from trek_lab.calibration import (
    ControlPoint, LocalPlane, Status, calibrate, fit_similarity, hull_coverage,
)

# A synthetic 4000x3000 px map near Kedarkantha (Uttarakhand): 2 m/px, rotated 15 deg.
W, H = 4000, 3000
ORIGIN = LocalPlane(31.02, 78.18)
_th = math.radians(15)
# pixel (x, y-down) -> (east, north) metres; the y flip makes north point up.
TRUE_M = np.array([
    [2 * math.cos(_th), 2 * math.sin(_th), -4000.0],
    [2 * math.sin(_th), -2 * math.cos(_th), 3000.0],
])


def true_latlon(x, y):
    e, n = TRUE_M[:, :2] @ [x, y] + TRUE_M[:, 2]
    return ORIGIN.to_latlon(e, n)


def cps(pixels, noise_m=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for x, y in pixels:
        e, n = TRUE_M[:, :2] @ [x, y] + TRUE_M[:, 2]
        e, n = e + rng.normal(0, noise_m), n + rng.normal(0, noise_m)
        lat, lon = ORIGIN.to_latlon(e, n)
        out.append(ControlPoint((x, y), lat, lon))
    return out


CORNERS = [(200, 200), (3800, 250), (3750, 2800), (250, 2750)]
SPREAD8 = CORNERS + [(2000, 300), (3700, 1500), (2000, 2700), (300, 1500)]


def haversine(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------- local plane

def test_local_plane_round_trip():
    lat, lon = ORIGIN.to_latlon(3500.0, -2200.0)
    e, n = ORIGIN.to_en(lat, lon)
    assert e == pytest.approx(3500.0, abs=0.01)
    assert n == pytest.approx(-2200.0, abs=0.01)


def test_local_plane_distance_matches_haversine():
    lat, lon = ORIGIN.to_latlon(4000.0, 3000.0)
    d = haversine(ORIGIN.origin_lat, ORIGIN.origin_lon, lat, lon)
    assert d == pytest.approx(5000.0, rel=0.003)  # haversine is a sphere; ellipsoid differs slightly


# ---------------------------------------------------------------- fitting

def test_similarity_handles_y_down_reflection():
    px = np.array([(200, 200), (3800, 2800)], float)
    en = px @ TRUE_M[:, :2].T + TRUE_M[:, 2]
    t = fit_similarity(px, en)
    probe = np.array([[1234.0, 987.0]])
    expected = probe @ TRUE_M[:, :2].T + TRUE_M[:, 2]
    assert np.allclose(t.apply(probe), expected, atol=1e-6)


def test_spread_exact_points_are_good():
    c = calibrate(cps(SPREAD8), (W, H))
    assert c.status is Status.GOOD
    assert c.rmse_m == pytest.approx(0.0, abs=0.05)
    assert c.transform.metres_per_pixel() == pytest.approx(2.0, rel=1e-3)


def test_four_points_never_good():
    # 4 points leave only 2 residual degrees of freedom: too little evidence for Good.
    c = calibrate(cps(CORNERS), (W, H))
    assert c.status is Status.APPROXIMATE


def test_round_trip_through_calibration():
    c = calibrate(cps(SPREAD8), (W, H))
    lat, lon = true_latlon(1500, 900)
    x, y = c.latlon_to_px(lat, lon)
    assert (x, y) == pytest.approx((1500, 900), abs=0.05)
    lat2, lon2 = c.px_to_latlon(x, y)
    assert haversine(lat, lon, lat2, lon2) < 0.1


# ---------------------------------------------------------------- status rules (18A.4)

def test_zero_and_one_point_insufficient():
    assert calibrate([], (W, H)).status is Status.INSUFFICIENT
    c = calibrate(cps([(100, 100)]), (W, H))
    assert c.status is Status.INSUFFICIENT and c.transform is None


def test_two_points_approximate_with_similarity():
    c = calibrate(cps([(200, 200), (3800, 2800)]), (W, H))
    assert c.status is Status.APPROXIMATE
    assert c.transform.kind == "similarity"
    assert c.resid_upper_m is None          # not measurable...
    assert c.prior_m == c.rmse_m > 0        # ...so estimated from assumed noise


def test_three_points_approximate_error_estimated_not_measured():
    c = calibrate(cps(CORNERS[:3]), (W, H))
    assert c.status is Status.APPROXIMATE
    assert c.resid_upper_m is None
    assert c.prior_m is not None


def test_clustered_points_never_good():
    cluster = [(1000, 1000), (1300, 1000), (1300, 1250), (1000, 1250), (1150, 1100)]
    c = calibrate(cps(cluster), (W, H))
    assert c.coverage < 0.15
    assert c.status is not Status.GOOD


def test_clustered_noise_is_amplified_by_leverage():
    # Same small noise: a tight cluster must report a much larger map-wide error than a
    # well-spread layout, because errors extrapolate far from the cluster.
    cluster = [(1000 + dx, 1000 + dy) for dx in (0, 150, 300) for dy in (0, 150, 300)]
    tight = calibrate(cps(cluster, noise_m=5, seed=3), (W, H))
    spread = calibrate(cps(SPREAD8 + [(2000, 1500)], noise_m=5, seed=3), (W, H))
    assert tight.rmse_m > 4 * spread.rmse_m
    assert tight.status is not Status.GOOD


def test_loo_method_still_available():
    c = calibrate(cps(SPREAD8, noise_m=5, seed=1), (W, H), method="loo")
    assert c.rmse_m == c.loo_rmse_m is not None


def test_noise_drives_status():
    good = calibrate(cps(SPREAD8, noise_m=5, seed=1), (W, H))
    approx = calibrate(cps(SPREAD8, noise_m=40, seed=1), (W, H))
    bad = calibrate(cps(SPREAD8, noise_m=200, seed=1), (W, H))
    assert good.status is Status.GOOD, good.rmse_m
    assert approx.status is Status.APPROXIMATE, approx.rmse_m
    assert bad.status is Status.INSUFFICIENT, bad.rmse_m


def test_collinear_points_insufficient():
    line = [(100, 100), (1000, 1000), (2000, 2000), (3000, 3000)]
    c = calibrate(cps(line), (W, H))
    assert c.status is Status.INSUFFICIENT


# ---------------------------------------------------------------- uncertainty

def test_uncertainty_combines_gps_and_calibration():
    c = calibrate(cps(SPREAD8, noise_m=5, seed=1), (W, H))
    r = c.uncertainty_m(10.0)
    assert r == pytest.approx(math.hypot(10.0, c.rmse_m))
    assert c.uncertainty_px(10.0) == pytest.approx(r / 2.0, rel=0.05)


def test_few_clustered_points_are_not_trusted():
    # Regression (found in the demo): 3 clustered points were shown with a flat 100 m
    # circle while the dot was really ~375 m off. Clustered few-point fits must report
    # large error and must not be shown as a usable position.
    tight = [(1800, 1400), (1950, 1420), (1880, 1560)]
    spread = [(200, 200), (3800, 300), (2000, 2800)]
    c_tight = calibrate(cps(tight), (W, H))
    c_spread = calibrate(cps(spread), (W, H))
    assert c_tight.rmse_m > 5 * c_spread.rmse_m
    assert c_tight.status is Status.INSUFFICIENT
    assert c_spread.status is Status.APPROXIMATE


def test_few_point_circle_contains_true_error():
    # Monte Carlo: with 3 noisy, spread points, the reported circle should contain the
    # true map-wide error in the large majority of runs.
    rng = np.random.default_rng(5)
    inside = 0
    for _ in range(200):
        pts = [(rng.uniform(200, 1300), rng.uniform(200, 1000)),
               (rng.uniform(2700, 3800), rng.uniform(200, 1000)),
               (rng.uniform(1300, 2700), rng.uniform(2000, 2800))]
        noisy = [ControlPoint((x + rng.normal(0, 10), y + rng.normal(0, 10)), *true_latlon(x, y))
                 for x, y in pts]
        c = calibrate(noisy, (W, H))
        gx, gy = np.meshgrid(np.linspace(0, W, 12), np.linspace(0, H, 12))
        err = [haversine(*true_latlon(x, y), *c.px_to_latlon(x, y)) for x, y in zip(gx.ravel(), gy.ravel())]
        inside += math.sqrt(np.mean(np.square(err))) <= c.rmse_m
    assert inside / 200 > 0.9


def test_hull_coverage_full_image():
    assert hull_coverage([(0, 0), (W, 0), (W, H), (0, H)], (W, H)) == pytest.approx(1.0)
