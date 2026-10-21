"""Image preparation: load any source, find the map in a photo, straighten it, extract its palette.

Pipeline stage 2-3 of the plan (section 6.1). The original is never modified; every
function returns new arrays plus the transform used, so results trace back to source pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORKING_MAX_EDGE = 8000  # decision 18A.2


# ---------------------------------------------------------------- loading

def load(path: str, page: int | None = None, max_edge: int = WORKING_MAX_EDGE) -> np.ndarray:
    """BGR uint8 image from JPEG/PNG/TIFF/PDF page (1-based), capped to max_edge."""
    if path.lower().endswith(".pdf"):
        import pypdfium2 as pdfium

        pg = pdfium.PdfDocument(path)[(page or 1) - 1]
        w, h = pg.get_size()
        scale = min(300 / 72, max_edge / max(w, h))  # 300 dpi, capped
        img = pg.render(scale=scale).to_numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if img.shape[2] == 4 else cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot read image: {path}")
    s = max_edge / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------- find + straighten

def _order(q: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    q = q.reshape(4, 2).astype(np.float32)
    s, d = q.sum(1), np.diff(q, axis=1).ravel()
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]], np.float32)


@dataclass
class QuadResult:
    quad: np.ndarray | None   # 4x2 corners in source pixels (TL, TR, BR, BL)
    area_frac: float          # quad area / image area
    method: str


def _angles_ok(q: np.ndarray, lo: float = 45, hi: float = 135) -> bool:
    q = q.reshape(4, 2).astype(float)
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if not lo <= np.degrees(np.arccos(np.clip(cos, -1, 1))) <= hi:
            return False
    return True


def _edge_support(edge_dist: np.ndarray, q: np.ndarray, tol: float = 3.0) -> float:
    """Fraction of the quad's perimeter lying within `tol` px of a real image edge."""
    q = q.reshape(4, 2).astype(float)
    pts = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        n = max(2, int(np.linalg.norm(b - a) / 2))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.append(a + t * (b - a))
    p = np.vstack(pts).round().astype(int)
    h, w = edge_dist.shape
    p[:, 0] = p[:, 0].clip(0, w - 1)
    p[:, 1] = p[:, 1].clip(0, h - 1)
    return float(np.mean(edge_dist[p[:, 1], p[:, 0]] <= tol))


def _candidate_quads(gray: np.ndarray, small: np.ndarray):
    """4-sided polygons from several segmentations: edges at a few thresholds, plus
    flat-colour regions (boards are usually one uniform panel)."""
    masks = []
    for lo, hi in ((30, 90), (60, 160)):
        e = cv2.Canny(gray, lo, hi)
        masks.append((f"canny{lo}", cv2.dilate(e, np.ones((3, 3), np.uint8))))
    masks.append(("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY, 51, -5)))
    lab = cv2.cvtColor(cv2.pyrMeanShiftFiltering(small, 12, 24), cv2.COLOR_BGR2LAB)
    for ch in range(3):
        _, m = cv2.threshold(lab[:, :, ch], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks += [(f"lab{ch}", m), (f"lab{ch}inv", 255 - m)]
    for name, m in masks:
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            peri = cv2.arcLength(c, True)
            for eps in (0.015, 0.03, 0.05):
                approx = cv2.approxPolyDP(c, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    yield name, approx
                    break


def find_map_quad(img: np.ndarray, min_area_frac: float = 0.02) -> QuadResult:
    """Best guess for the map/board outline in a photo.

    Candidates are convex 4-sided polygons from several segmentations. Each is scored by
    edge support (how much of its outline sits on a real edge) with a mild preference
    for larger quads. Returns quad=None when nothing plausible is found; the app then
    asks the user to drag the corners (the automatic result is only ever a suggestion)."""
    h, w = img.shape[:2]
    s = min(1.0, 1000 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    gray = cv2.bilateralFilter(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 9, 50, 50)
    edges = cv2.Canny(gray, 40, 120)
    edge_dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    area_total = small.shape[0] * small.shape[1]
    best, best_score, best_area, best_method = None, 0.0, 0.0, "none"
    for name, q in _candidate_quads(gray, small):
        a = cv2.contourArea(q)
        if not (min_area_frac * area_total <= a <= 0.95 * area_total) or not _angles_ok(q):
            continue
        support = _edge_support(edge_dist, q)
        if support < 0.6:
            continue
        score = support * (a / area_total) ** 0.25
        if score > best_score:
            best, best_score, best_area, best_method = q, score, a, f"{name} support {support:.0%}"
    if best is None:
        return QuadResult(None, 0.0, "none")
    return QuadResult(_order(best / s), best_area / area_total, best_method)


def rectify(img: np.ndarray, quad: np.ndarray, max_edge: int = WORKING_MAX_EDGE) -> tuple[np.ndarray, np.ndarray]:
    """Warp the quad to a flat rectangle. Returns (image, 3x3 homography source->rectified)."""
    q = _order(quad)
    wt = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    ht = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    s = min(1.0, max_edge / max(wt, ht))
    wt, ht = int(round(wt * s)), int(round(ht * s))
    dst = np.array([[0, 0], [wt - 1, 0], [wt - 1, ht - 1], [0, ht - 1]], np.float32)
    H = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, H, (wt, ht), flags=cv2.INTER_CUBIC), H


# ---------------------------------------------------------------- palette

@dataclass
class Palette:
    colors_bgr: np.ndarray    # k x 3 uint8
    share: np.ndarray         # fraction of pixels per colour
    labels: np.ndarray        # per-pixel colour index (same size as the input)


@dataclass
class InkLayers:
    ink_mask: np.ndarray      # bool: pixel is "ink" (line/text/symbol), not paper/fill
    palette: Palette          # clusters of ink pixels only; labels = -1 on paper
    paper: Palette            # clusters of the remaining (background/fill) pixels


def extract_ink_layers(img: np.ndarray, k_ink: int = 6, k_paper: int = 4,
                       bg_kernel: int = 21, threshold: float = 18.0, seed: int = 0) -> InkLayers:
    """Separate thin drawn features from the background they sit on, then cluster each.

    Whole-image colour clustering is dominated by large fills, so rivers, trails and
    contours (1-2% of pixels) never get their own colour. Here the local background is
    estimated with a median filter wider than any line; pixels that differ from it by
    more than `threshold` (CIELAB distance) are ink. Clustering only ink pixels gives
    each line colour its own layer."""
    k = bg_kernel | 1
    bg = cv2.medianBlur(img, k)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Chroma is measured relative to neutral (OpenCV stores a/b offset by 128).
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    chroma_bg = np.hypot(lab_bg[:, :, 1] - 128, lab_bg[:, :, 2] - 128)
    # Ink is drawn ON the paper: darker or more saturated than its surroundings. A plain
    # colour difference also fires along the edges of fills (green woodland meeting white
    # paper), which produces a fringe of false "lines" everywhere.
    darker = lab[:, :, 0] < lab_bg[:, :, 0] - 10
    more_saturated = chroma > chroma_bg + 12
    ink = (np.sqrt(((lab - lab_bg) ** 2).sum(-1)) > threshold) & (darker | more_saturated)
    # Two kinds of ink cluster separately so rare colours (blue water, red trails) are
    # not merged into the far more common black/grey text and linework.
    colour_ink = ink & (chroma >= 18)
    grey_ink = ink & ~colour_ink
    parts = []
    if colour_ink.sum() >= 50:
        parts.append((colour_ink, _hue_modes(img, lab, colour_ink)))
    if grey_ink.sum() >= 100:
        parts.append((grey_ink, extract_palette(img[grey_ink].reshape(-1, 1, 3), 3, seed=seed)))
    labels = np.full(img.shape[:2], -1, np.int32)
    colors, shares, offset = [], [], 0
    total = max(1, int(ink.sum()))
    for mask, pal in parts:
        labels[mask] = pal.labels.ravel() + offset
        colors.append(pal.colors_bgr)
        shares.append(pal.share * mask.sum() / total)
        offset += len(pal.colors_bgr)
    ink_pal = Palette(np.vstack(colors) if colors else np.zeros((0, 3), np.uint8),
                      np.concatenate(shares) if shares else np.zeros(0), labels)
    paper_px = img[~ink].reshape(-1, 1, 3)
    paper_pal = extract_palette(paper_px, k_paper, seed=seed)
    paper_labels = np.full(img.shape[:2], -1, np.int32)
    paper_labels[~ink] = paper_pal.labels.ravel()
    paper_pal.labels = paper_labels
    return InkLayers(ink, ink_pal, paper_pal)


def _hue_modes(img: np.ndarray, lab: np.ndarray, mask: np.ndarray,
               bins: int = 36, min_share: float = 0.001, max_modes: int = 6) -> Palette:
    """Cluster coloured ink by hue peaks rather than k-means, so a rare but distinct hue
    (a thin blue river among masses of brown contours) still gets its own layer."""
    a, b = lab[:, :, 1][mask] - 128, lab[:, :, 2][mask] - 128
    hue = (np.degrees(np.arctan2(b, a)) + 360) % 360
    hist = np.bincount((hue / (360 / bins)).astype(int) % bins, minlength=bins).astype(float)
    hist = np.convolve(np.r_[hist[-1], hist, hist[0]], [0.25, 0.5, 0.25], "valid")
    share = hist / hist.sum()
    peaks = [i for i in range(bins)
             if share[i] >= min_share and share[i] >= share[i - 1] and share[i] > share[(i + 1) % bins]]
    peaks = sorted(peaks, key=lambda i: -share[i])[:max_modes] or [int(share.argmax())]
    centres = np.array([(i + 0.5) * 360 / bins for i in peaks])
    d = np.abs((hue[:, None] - centres[None] + 180) % 360 - 180)
    lab_idx = d.argmin(1)
    px = img[mask]
    colors = np.array([np.median(px[lab_idx == j], axis=0) for j in range(len(peaks))], np.uint8)
    cnt = np.bincount(lab_idx, minlength=len(peaks)) / len(lab_idx)
    return Palette(colors, cnt, lab_idx)


def extract_palette(img: np.ndarray, k: int = 8, sample: int = 60000, seed: int = 0) -> Palette:
    """k-means in CIELAB (perceptual distances), fitted on a pixel sample then applied to
    the whole image. Colours are sorted by share, most common first."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    pick = lab[rng.choice(len(lab), min(sample, len(lab)), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(pick, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    # assign every pixel to its nearest centre, in chunks to bound memory
    labels = np.empty(len(lab), np.int32)
    for i in range(0, len(lab), 2_000_000):
        d = ((lab[i:i + 2_000_000, None, :] - centers[None]) ** 2).sum(-1)
        labels[i:i + 2_000_000] = d.argmin(1)
    share = np.bincount(labels, minlength=k) / len(labels)
    order = np.argsort(-share)
    remap = np.empty(k, np.int32)
    remap[order] = np.arange(k)
    centers_bgr = cv2.cvtColor(centers[order].reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR)[0]
    return Palette(centers_bgr, share[order], remap[labels].reshape(img.shape[:2]))
