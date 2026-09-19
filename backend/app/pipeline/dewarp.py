"""
LAYER 1 -- GEOMETRIC PREPROCESSING (DEWARPING)

TECH CHANGE (per systechchange.pdf): the prototype only had planar
homography, which cannot unwrap a label wrapped around a can/bottle --
a real, common failure mode this line of the doc calls out directly.
This module now offers TWO paths:

  1. warp_perspective()   -- unchanged 4-point homography (still correct
     and cheapest for the common flat cases: cartons, pouches, boxes,
     flat/large-radius bottle faces).
  2. cylindrical_unwarp()  -- NEW. Parametric cylindrical projection,
     following the formula in the doc:

        x_unwarped = R * arcsin((x - x_center) / R)

     R (the estimated cylinder radius in the mask's own pixel space) is
     derived from how much the mask's left/right edges bow inward relative
     to a straight vertical -- i.e. horizontal curvature -- rather than
     assumed. No trained mesh/TPS network (DocTr/UVDoc-style) is used here:
     that needs curved-label training data we don't have yet (Phase 2, same
     honesty policy as everywhere else in this codebase). Parametric
     cylindrical correction is a real, working, non-ML method that
     meaningfully improves OCR on genuinely curved surfaces without faking
     precision we don't have.

select_dewarp() picks between the two automatically from the segmentation
mask's curvature, so compliance_engine.py doesn't need to guess.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

# How much the mask must bow inward (as a fraction of its own width) before
# we treat the surface as "curved enough" to warrant cylindrical unwarp
# instead of homography. Conservative on purpose: homography on a flat
# label is strictly better than an unnecessary cylindrical projection.
CURVATURE_TRIGGER_RATIO = 0.06


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


def _estimate_curvature_ratio(mask: np.ndarray) -> float:
    """For each column of the mask, finds the top boundary y-position,
    then measures the sagitta (curve depth) of that top edge relative
    to a straight chord. This detects an upright cylinder's curvature.
    Returns 0.0 if the mask is empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0

    x_min, x_max = xs.min(), xs.max()
    if x_max <= x_min:
        return 0.0

    cols_to_sample = np.linspace(x_min, x_max, num=11).astype(int)
    top_edges = []
    for x in cols_to_sample:
        col_ys = ys[xs == x]
        if len(col_ys) > 0:
            top_edges.append((x, col_ys.min()))

    if len(top_edges) < 5:
        return 0.0

    # Calculate sagitta (curve depth) of the top edge
    x_start, y_start = top_edges[0]
    x_end, y_end = top_edges[-1]
    x_mid, y_mid = top_edges[len(top_edges) // 2]

    if x_end - x_start <= 0:
        return 0.0

    # Calculate where the straight line (chord) would be at the midpoint X
    chord_y_at_mid = y_start + (y_end - y_start) * ((x_mid - x_start) / (x_end - x_start))
    
    # Sagitta is the absolute vertical distance from the curve to the chord
    sagitta = abs(y_mid - chord_y_at_mid)
    width = float(x_end - x_start)

    bow_ratio = sagitta / width
    return max(0.0, float(bow_ratio))

def estimate_cylinder_radius_px(mask: np.ndarray) -> Optional[float]:
    """Derives R (pixels) from the mask's horizontal curvature. Returns
    None when the mask doesn't show meaningful curvature (caller should use
    homography instead)."""
    curvature = _estimate_curvature_ratio(mask)
    if curvature < CURVATURE_TRIGGER_RATIO:
        return None

    ys, xs = np.where(mask > 0)
    width_px = float(xs.max() - xs.min())
    if width_px <= 0:
        return None

    # A bow ratio of `curvature` over a chord of `width_px` implies an arc
    # whose sagitta (mid-chord depth) is curvature * width_px / 2. Standard
    # sagitta-chord-radius relation: R = (sagitta^2 + (chord/2)^2) / (2*sagitta).
    half_chord = width_px / 2.0
    sagitta = max(curvature * width_px / 2.0, 1e-3)
    radius = (sagitta ** 2 + half_chord ** 2) / (2 * sagitta)
    return radius


def cylindrical_unwarp(image_bgr: np.ndarray, mask: np.ndarray, out_width: int = 1200) -> Optional[np.ndarray]:
    """Projects the masked label region onto a flattened cylindrical
    surface per x_unwarped = R * arcsin((x - x_center) / R). Returns None
    if curvature isn't significant enough to warrant it -- caller should
    fall back to warp_perspective() in that case."""
    radius = estimate_cylinder_radius_px(mask)
    if radius is None:
        return None

    ys, xs = np.where(mask > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    crop = image_bgr[y_min:y_max, x_min:x_max]
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None

    x_center = cw / 2.0
    r = min(radius, cw)  # clamp: a degenerate radius estimate shouldn't blow up the map

    # Build the remap: for each destination column, find the source column
    # on the curved surface that maps to it (inverse of the forward formula,
    # since cv2.remap wants "where does this output pixel come from").
    out_h = int(round(out_width * (ch / max(cw, 1))))
    dst_xs = np.linspace(-cw / 2.0, cw / 2.0, out_width)
    # Inverse of x_unwarped = R*arcsin(x/R)  =>  x = R*sin(x_unwarped/R)
    src_xs = r * np.sin(np.clip(dst_xs / r, -1.0, 1.0)) + x_center
    src_xs = np.clip(src_xs, 0, cw - 1)

    map_x = np.tile(src_xs.astype(np.float32), (out_h, 1))
    map_y = np.repeat(np.linspace(0, ch - 1, out_h).astype(np.float32), out_width).reshape(out_h, out_width)

    unwarped = cv2.remap(crop, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return unwarped


def select_dewarp(
    image_bgr: np.ndarray,
    quad: List[Tuple[float, float]],
    mask: Optional[np.ndarray],
    out_width: int = 1200,
) -> Tuple[np.ndarray, str]:
    """Single entry point compliance_engine.py calls: picks cylindrical
    unwarp when the mask shows real curvature, otherwise the existing
    homography path. Always returns a usable image -- cylindrical_unwarp
    failing/returning None silently falls back to homography rather than
    breaking the pipeline."""
    if mask is not None:
        result = cylindrical_unwarp(image_bgr, mask, out_width=out_width)
        if result is not None:
            return result, "cylindrical"

    return warp_perspective(image_bgr, quad, out_width=out_width), "homography"


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
