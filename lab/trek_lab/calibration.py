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

Quality
-------
Error is the leave-one-out RMSE in metres: refit without each point and measure how
far that point lands from where it should. With exactly the minimum number of points
the fit is always perfect, so the error is unmeasurable (returned as None).
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
GOOD_MIN_POINTS = 4
GOOD_MIN_HULL_COVERAGE = 0.5


# ---------------------------------------------------------------- local tangent plane

def _ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
    return np.array([
        n * math.cos(lat) * math.cos(lon),
        n * math.cos(lat) * math.sin(lon),
        n * (1 - _E2) * math.sin(lat),
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

    def to_en(self, lat: float, lon: float) -> tuple[float, float]:
        d = _ecef(lat, lon) - _ecef(self.origin_lat, self.origin_lon)
        e, n, _ = self._rot() @ d
        return float(e), float(n)

    def to_latlon(self, east: float, north: float) -> tuple[float, float]:
        # Invert by iterating to_en; converges in a few steps for local distances.
        lat, lon = self.origin_lat, self.origin_lon
        for _ in range(6):
            e, n = self.to_en(lat, lon)
            m_per_deg_lat = math.radians(1) * _A
            m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat))
            lat += (north - n) / m_per_deg_lat
            lon += (east - e) / m_per_deg_lon
        return lat, lon


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
        """Radius to draw around the dot: sqrt(gps^2 + calibration^2). When calibration
        error is unmeasurable, assume the Approximate ceiling rather than zero."""
        calib = self.rmse_m if self.rmse_m is not None else APPROX_MAX_RMSE_M
        return math.hypot(gps_accuracy_m, calib)

    def uncertainty_px(self, gps_accuracy_m: float) -> float:
        if self.transform is None:
            raise ValueError("no transform: calibration is insufficient")
        return self.uncertainty_m(gps_accuracy_m) / self.transform.metres_per_pixel()


def classify(n: int, rmse_m: float | None, coverage: float) -> Status:
    if n < 2:
        return Status.INSUFFICIENT
    if rmse_m is not None and rmse_m > APPROX_MAX_RMSE_M:
        return Status.INSUFFICIENT
    if (n >= GOOD_MIN_POINTS and rmse_m is not None and rmse_m <= GOOD_MAX_RMSE_M
            and coverage >= GOOD_MIN_HULL_COVERAGE):
        return Status.GOOD
    return Status.APPROXIMATE


def calibrate(points: list[ControlPoint], image_size: tuple[int, int]) -> Calibration:
    n = len(points)
    if n == 0:
        return Calibration(LocalPlane(0.0, 0.0), None, None, 0.0, Status.INSUFFICIENT, 0)
    plane = LocalPlane(
        float(np.mean([p.lat for p in points])),
        float(np.mean([p.lon for p in points])),
    )
    px = np.array([p.px for p in points], float)
    en = np.array([plane.to_en(p.lat, p.lon) for p in points], float)
    transform = None
    if n >= 2:
        try:
            transform = fit(px, en)
        except ValueError:
            transform = None
    rmse = loo_rmse(px, en) if transform is not None else None
    coverage = hull_coverage(px, image_size)
    status = classify(n, rmse, coverage) if transform is not None else Status.INSUFFICIENT
    return Calibration(plane, transform, rmse, coverage, status, n)
