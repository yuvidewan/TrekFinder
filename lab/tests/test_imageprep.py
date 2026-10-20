import cv2
import numpy as np

from trek_lab.imageprep import extract_ink_layers, find_map_quad, rectify


def _synthetic_topo():
    """Cream paper, green fill patch, many brown contours, one thin blue river."""
    img = np.full((600, 600, 3), (225, 240, 245), np.uint8)       # cream (BGR)
    cv2.rectangle(img, (0, 300), (600, 600), (190, 235, 205), -1)  # green fill
    for r in range(40, 600, 14):
        cv2.circle(img, (300, 300), r, (60, 90, 160), 1)            # brown contours
    cv2.line(img, (0, 120), (600, 470), (200, 120, 40), 2)          # blue river
    return img


def test_rare_blue_river_gets_its_own_layer():
    layers = extract_ink_layers(_synthetic_topo())
    pal = layers.palette
    blues = [i for i, c in enumerate(pal.colors_bgr) if c[0] > 150 and c[2] < 100]
    assert blues, f"no blue layer among {pal.colors_bgr.tolist()}"
    river = pal.labels == blues[0]
    # the river layer is a thin line, not the fill
    assert 0.002 < river.mean() < 0.02


def test_fill_is_paper_not_ink():
    layers = extract_ink_layers(_synthetic_topo())
    assert not layers.ink_mask[450:460, 50:60].any() or layers.ink_mask[450:460, 50:60].mean() < 0.2


def test_finds_and_straightens_a_board():
    photo = np.full((800, 1200, 3), (60, 110, 70), np.uint8)       # foliage-ish
    rng = np.random.default_rng(0)
    photo = cv2.add(photo, rng.integers(0, 40, photo.shape, dtype=np.uint8))
    board = np.array([[300, 180], [950, 230], [900, 640], [260, 600]], np.int32)
    cv2.fillPoly(photo, [board], (235, 235, 235))
    q = find_map_quad(photo)
    assert q.quad is not None
    assert np.abs(q.quad - board).max() < 15
    flat, H = rectify(photo, q.quad)
    assert flat.shape[1] > flat.shape[0]  # landscape board stays landscape
