"""
LAYER 0 (NOVELTY #1) -- BARCODE-AS-SELF-CALIBRATING-RULER

MIGRATED per systechchange.pdf "Technology Replacement Matrix": pyzbar/zbar
-> zxing-cpp. pyzbar's 1D scanline thresholding fails on tilt > ~15deg,
curved surfaces (bottles/cans), low contrast, or blur -- all extremely
common in real phone photos of packaging. zxing-cpp ships modern 2D
binarizers (HybridBinarizer) that are far more tolerant of these
conditions, and returns the four corner points of the detected symbol
directly (no separate bounding-rect step needed).

CRITICAL ORDERING RULE (unchanged from before, now doubly important):
this must run on the RAW, un-warped, native-resolution image. CLAHE and
homography both interpolate pixel boundaries and destroy the sharp
black/white module-width ratios the decoder depends on. compliance_engine.py
already does this correctly (calibrates on `image_bgr` first, only falls
back to `enhanced` if nothing found there) -- do not change that ordering
when wiring this in.

Nominal EAN-13/UPC-A physical bar-pattern width used for scaling: 31.35mm
(the bar pattern itself, excluding quiet zones) -- per systechchange.pdf's
formula. This differs from the old pyzbar version's 37.29mm figure (which
included the printed quiet-zone margins); using the wrong constant here
will silently mis-scale every font-size measurement downstream, so if you
compare against old saved scans expect calibration_factor to shift.
"""

from typing import Optional
import numpy as np
import cv2
import zxingcpp

from app.pipeline.types import Calibration, BBox

# Bar-pattern-only nominal widths (mm), per systechchange.pdf.
NOMINAL_WIDTH_MM = {
    "EAN13": 31.35,
    "UPCA": 31.35,
    "EAN8": 22.85,  # proportional EAN-8 bar-pattern estimate
}

SYMBOLOGY_MAP = {
    "EAN13": "EAN_13",
    "UPCA": "UPC_A",
    "EAN8": "EAN_8",
}

ACCEPTED_FORMATS = {
    zxingcpp.BarcodeFormat.EAN13: "EAN13",
    zxingcpp.BarcodeFormat.UPCA: "UPCA",
    zxingcpp.BarcodeFormat.EAN8: "EAN8",
}


def _pt(p) -> np.ndarray:
    return np.array([p.x, p.y], dtype=np.float64)


def detect_barcode_calibration(image_bgr: np.ndarray) -> Calibration:
    """Runs zxing-cpp on the given image. MUST be called on the raw/original
    frame -- see module docstring. compliance_engine.py's `_calibrate()`
    closure already handles the "try raw, fall back to enhanced" logic."""
    try:
        results = zxingcpp.read_barcodes(image_bgr)
    except Exception:
        return Calibration(found=False, symbology="NONE")

    candidates = [r for r in results if r.valid and r.format in ACCEPTED_FORMATS]
    if not candidates:
        return Calibration(found=False, symbology="NONE")

    # Prefer the largest detected symbol (most likely the primary, in-focus
    # barcode rather than a small secondary/promo code elsewhere on pack).
    def _area(r):
        pos = r.position
        w = max(cv2.norm(_pt(pos.top_left) - _pt(pos.top_right)),
                 cv2.norm(_pt(pos.bottom_left) - _pt(pos.bottom_right)))
        h = max(cv2.norm(_pt(pos.top_left) - _pt(pos.bottom_left)),
                 cv2.norm(_pt(pos.top_right) - _pt(pos.bottom_right)))
        return w * h

    best = max(candidates, key=_area)
    symbology = ACCEPTED_FORMATS[best.format]
    pos = best.position

    width_px = float(max(
        cv2.norm(_pt(pos.top_left) - _pt(pos.top_right)),
        cv2.norm(_pt(pos.bottom_left) - _pt(pos.bottom_right)),
    ))
    height_px = float(max(
        cv2.norm(_pt(pos.top_left) - _pt(pos.bottom_left)),
        cv2.norm(_pt(pos.top_right) - _pt(pos.bottom_right)),
    ))

    physical_width_mm = NOMINAL_WIDTH_MM.get(symbology, 31.35)
    px_per_mm = width_px / physical_width_mm

    xs = [pos.top_left.x, pos.top_right.x, pos.bottom_left.x, pos.bottom_right.x]
    ys = [pos.top_left.y, pos.top_right.y, pos.bottom_left.y, pos.bottom_right.y]
    bbox = BBox(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))

    return Calibration(
        found=True,
        symbology=SYMBOLOGY_MAP.get(symbology, symbology),
        raw_value=best.text,
        pixel_width=width_px,
        physical_width_mm=physical_width_mm,
        px_per_mm=px_per_mm,
        bbox=bbox,
    )


def manual_calibration(p1: tuple, p2: tuple, known_distance_mm: float) -> Calibration:
    """Unchanged fallback for when no barcode is visible/decodable: the user
    taps two points a known real-world distance apart."""
    pixel_width = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return Calibration(
        found=True,
        symbology="MANUAL",
        pixel_width=pixel_width,
        physical_width_mm=known_distance_mm,
        px_per_mm=pixel_width / known_distance_mm,
    )
