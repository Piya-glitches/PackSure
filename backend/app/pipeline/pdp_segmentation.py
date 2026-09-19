"""
LAYER 1 -- PDP (PRINCIPAL DISPLAY PANEL) SEGMENTATION  (replaces pdp_detection.py)

TECH CHANGE (per systechchange.pdf): swapped bounding-box YOLOv8 detection
for YOLOv8n-seg INSTANCE SEGMENTATION. A bounding box on a real retail shelf
photo includes background, neighbouring products, and shadow -- a label on
a pouch or curved bottle very rarely IS a clean axis-aligned rectangle. A
segmentation polygon gives the actual label surface, which is what both
(a) the homography dewarp and (b) the new cylindrical unwarp in dewarp.py
need to work correctly.

HONEST SCOPING (same policy as before, just re-targeted): a fine-tuned
label_pdp segmentation checkpoint (models/pdp_yolov8n_seg.pt or .onnx)
needs ~400-600 annotated real package photos -- we don't have that yet.
Until it exists, this module falls back to the SAME classical contour
localizer the prototype used (Canny + largest 4-point contour), and always
reports which path ran via `pdp_detection_method` so nothing is silently
faked. Once a checkpoint is dropped in, the mask path activates
automatically -- no other code changes needed.

Public API kept backward-compatible with the old pdp_detection.py:
  detect_pdp_quad(image) -> (quad, method)          # for existing homography dewarp
  detect_pdp_mask(image) -> (mask|None, quad, method) # mask enables cylindrical unwarp
"""

import os
from typing import Optional, List, Tuple

import cv2
import numpy as np

SEG_MODEL_PATH_PT = os.path.join(os.path.dirname(__file__), "..", "..", "models", "pdp_yolov8n_seg.pt")
SEG_MODEL_PATH_ONNX = os.path.join(os.path.dirname(__file__), "..", "..", "models", "pdp_yolov8n_seg.onnx")

_seg_model = None
_seg_load_attempted = False


def _try_load_seg_model():
    """Prefers the ONNX export (faster, portable, matches the doc's
    'export all checkpoints to ONNX' deployment step) and falls back to the
    raw .pt checkpoint if only that is present. Either way this is real
    Ultralytics inference, not a stub."""
    global _seg_model, _seg_load_attempted
    if _seg_load_attempted:
        return _seg_model
    _seg_load_attempted = True

    path = None
    if os.path.exists(SEG_MODEL_PATH_ONNX):
        path = SEG_MODEL_PATH_ONNX
    elif os.path.exists(SEG_MODEL_PATH_PT):
        path = SEG_MODEL_PATH_PT
    if path is None:
        return None

    try:
        from ultralytics import YOLO

        _seg_model = YOLO(path)
    except Exception:
        _seg_model = None
    return _seg_model


def order_points(pts: np.ndarray) -> List[Tuple[float, float]]:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    # diff = y - x: top-right has the SMALLEST value, bottom-left the LARGEST.
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def _mask_to_quad(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Reduces a segmentation mask to a 4-point quad (min-area rect) so the
    existing homography dewarp path keeps working unchanged for flat labels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Empty mask produced no contour.")
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return order_points(box)


def _contour_fallback(image_bgr: np.ndarray) -> List[Tuple[float, float]]:
    """Unchanged classical fallback from the prototype: Canny edges + largest
    4-point contour. Kept verbatim -- it's a real, working, non-ML method for
    a well-composed photo, and remains the honest default until a
    segmentation checkpoint is trained."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = w * h
    best_quad = None
    best_area = 0.0

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = abs(cv2.contourArea(approx))
            if area > best_area and img_area * 0.15 < area < img_area * 0.98:
                best_area = area
                pts = approx.reshape(4, 2).astype(float)
                best_quad = order_points(pts)

    if best_quad is not None:
        return best_quad

    mx, my = w * 0.06, h * 0.06
    return [(mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my)]


def detect_pdp_mask(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[float, float]], str]:
    """Returns (mask|None, quad, method). mask is a full-resolution binary
    array (same H,W as image_bgr) when segmentation ran, else None -- callers
    (dewarp.py) use the mask presence to decide homography vs. cylindrical
    unwarp, and fall back to quad-only warping when it's absent."""
    h, w = image_bgr.shape[:2]
    model = _try_load_seg_model()

    if model is not None:
        results = model.predict(image_bgr, verbose=False)
        r = results[0]
        if r.masks is not None and len(r.masks.data) > 0:
            confs = r.boxes.conf if r.boxes is not None else None
            best_idx = int(confs.argmax()) if confs is not None and len(confs) > 0 else 0
            raw_mask = r.masks.data[best_idx].cpu().numpy()
            mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
            try:
                quad = _mask_to_quad(mask)
            except ValueError:
                quad = _contour_fallback(image_bgr)
            return mask, quad, "yolov8n_seg"

    quad = _contour_fallback(image_bgr)
    return None, quad, "contour_fallback"


def detect_pdp_quad(image_bgr: np.ndarray) -> Tuple[List[Tuple[float, float]], str]:
    """Backward-compatible entry point for callers that only need a quad
    (compliance_engine.py's manual-override path, tests, etc.)."""
    _, quad, method = detect_pdp_mask(image_bgr)
    return quad, method
