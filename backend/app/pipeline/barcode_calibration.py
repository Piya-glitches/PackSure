"""
LAYER 0 (NOVELTY #1) -- BARCODE-AS-SELF-CALIBRATING-RULER

Detects an EAN-13/UPC-A/EAN-8 barcode via pyzbar (exactly as specified:
"Detect EAN-13/UPC barcode via pyzbar/ZXing, use its ISO-standard physical
width as an in-frame reference scale"), measures its pixel width, and
divides the GS1-standardised physical width by that pixel width to get a
pixels-per-millimetre calibration factor for THIS specific photo.

This is what turns every barcode already on a package into a built-in
ruler -- no coin, no printed reference card, no special hardware. Every
industrial vision system in the market assumes a fixed, calibrated camera
rig; we have no such luxury with a phone photo, so this is the layer that
makes real physical measurement (Layer 4's font-size validator) possible
at all in an uncontrolled environment.

Nominal physical widths at 100% GS1 magnification (symbol body incl. guard
bars): EAN-13 / UPC-A = 37.29mm, EAN-8 = 26.73mm. Real packages print at
80%-200% magnification, so this is an approximation unless a fine-tuned
scale factor is supplied -- surfaced honestly rather than claiming false
precision (see manual calibration fallback below).
"""

from typing import Optional
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol

from app.pipeline.types import Calibration, BBox

NOMINAL_WIDTH_MM = {
    "EAN13": 37.29,
    "UPCA": 37.29,
    "EAN8": 26.73,
}

SYMBOLOGY_MAP = {
    "EAN13": "EAN_13",
    "UPCA": "UPC_A",
    "EAN8": "EAN_8",
}


def detect_barcode_calibration(image_bgr: np.ndarray) -> Calibration:
    results = decode(image_bgr, symbols=[ZBarSymbol.EAN13, ZBarSymbol.UPCA, ZBarSymbol.EAN8])

    if not results:
        return Calibration(found=False, symbology="NONE")

    # Use the largest detected barcode (most likely to be the primary,
    # in-focus one rather than a tiny secondary code elsewhere on pack).
    best = max(results, key=lambda r: r.rect.width * r.rect.height)
    symbology = best.type  # "EAN13" | "UPCA" | "EAN8"

    physical_width_mm = NOMINAL_WIDTH_MM.get(symbology, 37.29)
    pixel_width = float(best.rect.width)
    px_per_mm = pixel_width / physical_width_mm

    bbox = BBox(x=float(best.rect.left), y=float(best.rect.top), w=pixel_width, h=float(best.rect.height))

    return Calibration(
        found=True,
        symbology=SYMBOLOGY_MAP.get(symbology, symbology),
        raw_value=best.data.decode("utf-8", errors="ignore"),
        pixel_width=pixel_width,
        physical_width_mm=physical_width_mm,
        px_per_mm=px_per_mm,
        bbox=bbox,
    )


def manual_calibration(p1: tuple, p2: tuple, known_distance_mm: float) -> Calibration:
    """Fallback for when no barcode is visible/decodable: the user taps two
    points a known real-world distance apart (e.g. a currency note edge,
    ID card, or ruler placed in-frame)."""
    pixel_width = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return Calibration(
        found=True,
        symbology="MANUAL",
        pixel_width=pixel_width,
        physical_width_mm=known_distance_mm,
        px_per_mm=pixel_width / known_distance_mm,
    )
