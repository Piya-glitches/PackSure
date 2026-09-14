"""
LAYER 1 -- GEOMETRIC PREPROCESSING (DEWARPING)

Per the architecture: "Homography via OpenCV (4-point transform). Handles
angled phone shots -- cheap, standard, essential." Takes the quad located
by pdp_detection.py (or a manual override sent from the frontend's
corner-drag editor) and warps it into a flat, fronto-parallel rectangle.

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied
afterward per Layer 2's "Low-res / glare handling" requirement, recovering
text on foil/glossy/low-contrast labels before OCR runs.

HONEST SCOPING: true cylindrical (curved-surface) dewarping for a label
wrapped around a bottle -- as in the RecRecNet/thin-plate-spline research
cited in the architecture -- requires a trained dense-correspondence
network, which is a Phase 2 item (needs training data of curved labels
with ground-truth flat counterparts). Planar homography correction here
still meaningfully improves OCR accuracy on the majority of labels
(cartons, pouches, boxes, flat/large-radius bottle faces), and we do not
fake curved dewarping with an untrained heuristic, since that would
silently degrade OCR accuracy rather than improve it.
"""

from typing import List, Tuple
import cv2
import numpy as np


def warp_perspective(image_bgr: np.ndarray, quad: List[Tuple[float, float]], out_width: int = 1200) -> np.ndarray:
    (tl, tr, br, bl) = quad

    width_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    width_bottom = np.hypot(br[0] - bl[0], br[1] - bl[1])
    height_left = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    height_right = np.hypot(br[0] - tr[0], br[1] - tr[1])

    out_w = out_width
    aspect = max(width_top, width_bottom) / max(height_left, height_right, 1.0)
    out_h = int(round(out_w / aspect)) if aspect > 0 else out_w

    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_pts = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image_bgr, m, (out_w, out_h))
    return warped


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
