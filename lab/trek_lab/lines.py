"""Line extraction: turn one ink layer (e.g. "the trail colour") into centrelines.

Pipeline stage 5 (geometry extraction). Steps:
1. drop compact blobs (letters, dots, symbols) that are not line-like;
2. bridge small gaps so dashed and dotted lines become continuous;
3. thin to 1-px centrelines and drop short fragments.
The result is a raster skeleton; vectorising it into polylines comes after.
"""

from __future__ import annotations

import cv2
import numpy as np


def drop_blobs(mask: np.ndarray, max_blob_px: int = 400, min_elongation: float = 2.5) -> np.ndarray:
    """Remove small compact components. A component is kept if it is large, or if it is
    elongated (a dash or a line segment) rather than round/boxy (a letter or a dot)."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_blob_px:
            keep[i] = True
            continue
        ys, xs = np.nonzero(lab[stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT],
                                stats[i, cv2.CC_STAT_LEFT]:stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]] == i)
        if len(xs) < 3:
            continue
        cov = np.cov(np.vstack([xs, ys]).astype(float))
        ev = np.sort(np.linalg.eigvalsh(cov))
        keep[i] = ev[1] > (min_elongation ** 2) * max(ev[0], 0.25)
    return keep[lab]


def centrelines(mask: np.ndarray, gap_px: int = 5, min_len_px: int = 25,
                blob_px: int = 400) -> np.ndarray:
    """1-px skeleton of the line-like structures in `mask` (bool), gaps up to ~gap_px bridged."""
    m = drop_blobs(mask, max_blob_px=blob_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1, 2 * gap_px + 1))
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)
    sk = cv2.ximgproc.thinning(m, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((sk > 0).astype(np.uint8), connectivity=8)
    long_enough = np.zeros(n, bool)
    long_enough[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_len_px
    return long_enough[lab]
