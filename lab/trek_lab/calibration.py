"""Map calibration: relate image pixels to real-world positions (plan sections 7 and 18A.4).

Conventions
-----------
- Pixel coordinates are (x, y) with y pointing DOWN, as in images.
- Geographic positions are WGS84 (lat, lon) in degrees.
- All fitting happens in a local tangent plane (east, north) in metres, centred on a
  reference origin near the map. Fitting directly in lat/lon would distort scale.

Transform models
----------------
- 2 points  -> similarity (uniform scale, rotation, translation).
- 3+ points -> least-squares affine (adds independent x/y scale and skew).

Quality (revised in Phase 0, see experiments/calib_sim.py and calib_tune.py)
-------
The reported error is the 95% upper confidence bound of the map-wide RMS position
error, estimated from the affine fit's residuals and the leverage of every map point
(residual_error). It needs 4+ points; with fewer the error is unmeasurable (None).

Good = 5+ points, bound <= 25 m, hull coverage >= 15%. In simulation over 5 georeferenced
maps this rated ~4.5% of Goods worse than 25 m, the worst at 33 m. The plan v2 rule
(leave-one-out RMSE, 4 points, 50% coverage) was ~2x pessimistic for well-spread points,
under-reported error in 89% of clustered cases, and made Good unreachable in practice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

# WGS84 ellipsoid
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)

GOOD_MAX_RMSE_M = 25.0
APPROX_MAX_RMSE_M = 100.0
GOOD_MIN_POINTS = 5
GOOD_MIN_HULL_COVERAGE = 0.15
RESID_CONF = 0.95


# ---------------------------------------------------------------- local tangent plane

def _ecef(lat_deg, lon_deg) -> np.ndarray:
    """ECEF metres; accepts scalars or arrays, returns shape (3, ...)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.array([
        n * np.cos(lat) * np.cos(lon),
        n * np.cos(lat) * np.sin(lon),
        n * (1 - _E2) * np.sin(lat),
    ])


@dataclass(frozen=True)
class LocalPlane:
    """East/north metres around an origin. Exact ENU rotation at altitude 0, with the
    vertical component dropped; accurate to well under a metre for map-sized areas."""

    origin_lat: float
    origin_lon: float

    def _rot(self) -> np.ndarray:
        lat, lon = math.radians(self.origin_lat), math.radians(self.origin_lon)
        sl, cl, sp, cp = math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)
        return np.array([
            [-sl, cl, 0.0],
            [-sp * cl, -sp * sl, cp],
            [cp * cl, cp * sl, sp],
        ])

    def en_array(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        d = _ecef(lat, lon) - _ecef(self.origin_lat, self.origin_lon).reshape((3,) + (1,) * lat.ndim)
        e, n, _ = np.tensordot(self._rot(), d, axes=1)
        return e, n

    def latlon_array(self, east, north) -> tuple[np.ndarray, np.ndarray]:
        # Invert by iterating en_array; converges in a few steps for local distances.
        east, north = np.asarray(east, float), np.asarray(north, float)
        lat = np.full_like(east, self.origin_lat)
        lon = np.full_like(east, self.origin_lon)
        m_per_deg_lat = math.radians(1) * _A
        for _ in range(6):
            e, n = self.en_array(lat, lon)
            lat = lat + (north - n) / m_per_deg_lat
            lon = lon + (east - e) / (m_per_deg_lat * np.cos(np.radians(lat)))
        return lat, lon

    def to_en(self, lat: float, lon: float) -> tuple[float, float]:
        e, n = self.en_array(lat, lon)
        return float(e), float(n)

    def to_latlon(self, east: float, north: float) -> tuple[float, float]:
        lat, lon = self.latlon_array(east, north)
        return float(lat), float(lon)


# ---------------------------------------------------------------- transforms

@dataclass(frozen=True)
class Transform:
    """Pixel (x, y-down) -> local plane (east, north) metres, as a 2x3 affine matrix."""

    kind: str  # "similarity" | "affine"
    m: np.ndarray  # shape (2, 3)

    def apply(self, pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        return p @ self.m[:, :2].T + self.m[:, 2]

    def inverse_apply(self, en) -> np.ndarray:
        q = np.atleast_2d(np.asarray(en, dtype=float))
        inv = np.linalg.inv(self.m[:, :2])
        return (q - self.m[:, 2]) @ inv.T

    def metres_per_pixel(self) -> float:
        """Geometric-mean scale; used to turn metre uncertainties into pixel radii."""
        return math.sqrt(abs(np.linalg.det(self.m[:, :2])))


def fit_similarity(px, en) -> Transform:
    """Least-squares similarity. Pixel y points down and north points up, so the true
    mapping includes a reflection; we flip y first, fit a rotation+scale with complex
    numbers, then fold the flip back into the matrix."""
    px, en = np.asarray(px, float), np.asarray(en, float)
    if len(px) < 2:
        raise ValueError("similarity needs at least 2 points")
    z = px[:, 0] - 1j * px[:, 1]          # flipped pixel coords as complex
    w = en[:, 0] + 1j * en[:, 1]
    zc, wc = z - z.mean(), w - w.mean()
    denom = np.vdot(zc, zc).real
    if denom == 0:
        raise ValueError("control points are coincident")
    a = np.vdot(zc, wc) / denom           # complex scale*rotation
    b = w.mean() - a * z.mean()
    # w = a * (x - i y) + b  ->  east/north rows in terms of x, y
    m = np.array([
        [a.real, a.imag, b.real],
        [a.imag, -a.real, b.imag],
    ])
    return Transform("similarity", m)


def fit_affine(px, en) -> Transform:
    px, en = np.asarray(px, float), np.asarray(en, float)
    if len(px) < 3:
        raise ValueError("affine needs at least 3 points")
    A = np.column_stack([px, np.ones(len(px))])
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("control points are collinear")
    sol, *_ = np.linalg.lstsq(A, en, rcond=None)
    return Transform("affine", sol.T)


def fit(px, en) -> Transform:
    return fit_similarity(px, en) if len(px) == 2 else fit_affine(px, en)


def loo_rmse(px, en) -> float | None:
    """Leave-one-out RMSE in metres, or None when it cannot be measured. Needs 4+ points:
    each refit then uses 3+ points for an affine, which predicts the left-out point.
    (With 3 points we could refit similarities on pairs, but that would measure the
    similarity model's error, not the affine's, so the plan treats it as unmeasured.)"""
    px, en = np.asarray(px, float), np.asarray(en, float)
    n = len(px)
    if n <= 3:
        return None
    errs = []
    for i in range(n):
        keep = np.arange(n) != i
        try:
            t = fit(px[keep], en[keep])
        except ValueError:
            return None
        errs.append(np.sum((t.apply(px[i])[0] - en[i]) ** 2))
    return float(math.sqrt(np.mean(errs)))


def _chi2_quantile(p: float, dof: int) -> float:
    """Wilson–Hilferty approximation; slightly conservative for tiny dof, no SciPy needed
    (keeps the C++ port dependency-free)."""
    # inverse normal via Acklam-free rational approx is overkill; we only need p=0.1/0.05
    z = {0.05: -1.6449, 0.1: -1.2816, 0.2: -0.8416}[p]
    k = float(dof)
    return k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3


def residual_error(px, en, image_size: tuple[int, int], conf: float | None = None,
                   grid: int = 15) -> tuple[float, float] | None:
    """Map-wide RMS position error of an affine fit, estimated from its residuals.

    sigma^2 = RSS / (2n - 6) is the per-axis noise of the control points. The fitted
    transform's error at map point g has variance sigma^2 * g'(A'A)^-1 g per axis
    (leverage), so averaging over a grid covering the image gives the expected RMS error
    of the dot anywhere on the map. Clustered points produce high leverage far away, so
    this penalises poor spread on its own.

    Returns (estimate, upper bound at `conf`), or None for fewer than 4 points (no
    residual degrees of freedom)."""
    px, en = np.asarray(px, float), np.asarray(en, float)
    n = len(px)
    if n < 4:
        return None
    A = np.column_stack([px, np.ones(n)])
    try:
        ata_inv = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return None
    sol, *_ = np.linalg.lstsq(A, en, rcond=None)
    rss = float(np.sum((A @ sol - en) ** 2))
    dof = 2 * n - 6
    w, h = image_size
    gx, gy = np.meshgrid(np.linspace(0, w, grid), np.linspace(0, h, grid))
    G = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    lev = float(np.mean(np.einsum("ij,jk,ik->i", G, ata_inv, G)))
    sigma2 = rss / dof
    conf = RESID_CONF if conf is None else conf
    sigma2_up = rss / _chi2_quantile(round(1 - conf, 2), dof)
    return math.sqrt(2 * sigma2 * lev), math.sqrt(2 * sigma2_up * lev)


PRIOR_GPS_M = 10.0   # assumed error of an "I'm here" fix / entered coordinate
PRIOR_TAP_PX = 10.0  # assumed finger-tap error on the map, in image pixels
PRIOR_SAFETY = 2.0   # these are guesses, not measurements: stay cautious


def prior_error(px, transform: "Transform", image_size: tuple[int, int], grid: int = 15) -> float:
    """Map-wide RMS error expected from assumed tap/GPS noise, for fits with too few points
    to measure their own error (2-3 points). Clustered points extrapolate badly, so the
    leverage term grows with distance from them, exactly as in residual_error."""
    px = np.asarray(px, float)
    n = len(px)
    mpp = transform.metres_per_pixel()
    sigma = math.hypot(PRIOR_GPS_M, PRIOR_TAP_PX * mpp)
    w, h = image_size
    gx, gy = np.meshgrid(np.linspace(0, w, grid), np.linspace(0, h, grid))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    if n == 2:
        # similarity: one fixed point (midpoint) plus rotation/scale from the baseline;
        # error ~ sigma * sqrt(1/2 + 2 d^2 / L^2) at distance d from the midpoint
        c, L = px.mean(0), np.linalg.norm(px[1] - px[0])
        d2 = ((G - c) ** 2).sum(1)
        lev = 0.5 + 2 * d2 / max(L, 1.0) ** 2
        rms = sigma * math.sqrt(float(np.mean(lev)))
    else:
        A = np.column_stack([px, np.ones(n)])
        Gh = np.column_stack([G, np.ones(len(G))])
        lev = np.einsum("ij,jk,ik->i", Gh, np.linalg.inv(A.T @ A), Gh)
        rms = sigma * math.sqrt(2 * float(np.mean(lev)))
    return PRIOR_SAFETY * rms


# ---------------------------------------------------------------- coverage

def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def hull_coverage(px, image_size: tuple[int, int]) -> float:
    """Fraction of the image area enclosed by the control points' convex hull."""
    hull = _convex_hull(np.asarray(px, float))
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    w, h = image_size
    return float(area / (w * h))


# ---------------------------------------------------------------- calibration result

class Status(str, Enum):
    GOOD = "good"
    APPROXIMATE = "approximate"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class ControlPoint:
    px: tuple[float, float]       # pixel (x, y-down)
    lat: float
    lon: float
    source: str = "landmark"      # "grid" | "landmark" | "im_here"


@dataclass(frozen=True)
class Calibration:
    plane: LocalPlane
    transform: Transform | None
    rmse_m: float | None
    coverage: float
    status: Status
    n_points: int
    # Diagnostics: both error estimators, whichever `method` was used for rmse_m.
    loo_rmse_m: float | None = None
    resid_rmse_m: float | None = None
    resid_upper_m: float | None = None
    # Set when rmse_m came from assumed tap/GPS noise (2-3 points), not measurement.
    prior_m: float | None = None

    def latlon_to_px(self, lat: float, lon: float) -> tuple[float, float]:
        if self.transform is None:
            raise ValueError("no transform: calibration is insufficient")
        e, n = self.plane.to_en(lat, lon)
        x, y = self.transform.inverse_apply([e, n])[0]
        return float(x), float(y)

    def px_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        if self.transform is None:
            raise ValueError("no transform: calibration is insufficient")
        e, n = self.transform.apply([x, y])[0]
        return self.plane.to_latlon(e, n)

    def uncertainty_m(self, gps_accuracy_m: float) -> float:
        """Radius to draw around the dot: sqrt(gps^2 + calibration^2). With 2-3 points the
        calibration term is the prior-based estimate (see prior_error); the 100 m
        fallback only applies if no estimate exists at all."""
        calib = self.rmse_m if self.rmse_m is not None else APPROX_MAX_RMSE_M
        return math.hypot(gps_accuracy_m, calib)

    def uncertainty_px(self, gps_accuracy_m: float) -> float:
        if self.transform is None:
            raise ValueError("no transform: calibration is insufficient")
        return self.uncertainty_m(gps_accuracy_m) / self.transform.metres_per_pixel()


def classify(n: int, rmse_m: float | None, coverage: float, measured: bool = True) -> Status:
    """Error estimated from assumed noise (2-3 points) is deliberately pessimistic, so it
    only rules a calibration out when clearly hopeless (> 2x the Approximate ceiling);
    the wide circle still tells the user how unsure the position is."""
    if n < 2:
        return Status.INSUFFICIENT
    ceiling = APPROX_MAX_RMSE_M if measured else 2 * APPROX_MAX_RMSE_M
    if rmse_m is not None and rmse_m > ceiling:
        return Status.INSUFFICIENT
    if (n >= GOOD_MIN_POINTS and rmse_m is not None and rmse_m <= GOOD_MAX_RMSE_M
            and coverage >= GOOD_MIN_HULL_COVERAGE):
        return Status.GOOD
    return Status.APPROXIMATE


def calibrate(points: list[ControlPoint], image_size: tuple[int, int],
              method: str = "residual") -> Calibration:
    """method: "loo" (plan v2: leave-one-out RMSE) or "residual" (upper confidence bound
    of the residual-based map-wide error; see residual_error)."""
    n = len(points)
    if n == 0:
        return Calibration(LocalPlane(0.0, 0.0), None, None, 0.0, Status.INSUFFICIENT, 0)
    plane = LocalPlane(
        float(np.mean([p.lat for p in points])),
        float(np.mean([p.lon for p in points])),
    )
    px = np.array([p.px for p in points], float)
    e, nn = plane.en_array([p.lat for p in points], [p.lon for p in points])
    en = np.column_stack([e, nn])
    transform = None
    if n >= 2:
        try:
            transform = fit(px, en)
        except ValueError:
            transform = None
    coverage = hull_coverage(px, image_size)
    if transform is None:
        return Calibration(plane, None, None, coverage, Status.INSUFFICIENT, n)
    loo = loo_rmse(px, en)
    resid = residual_error(px, en, image_size)
    r_est, r_up = resid if resid else (None, None)
    rmse = loo if method == "loo" else r_up
    prior = None
    if rmse is None and method != "loo":
        # Too few points to measure error: estimate it from assumed noise and leverage.
        prior = prior_error(px, transform, image_size)
        rmse = prior
    status = classify(n, rmse, coverage, measured=prior is None)
    return Calibration(plane, transform, rmse, coverage, status, n, loo, r_est, r_up, prior)
