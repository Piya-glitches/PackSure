"""
LAYER 4 -- FONT-SIZE / READABILITY VALIDATOR

Uses Layer 0's barcode-derived px-per-mm calibration factor to convert a
detected field's pixel bounding-box height into real millimetres, then
compares against the LMPC minimum character-height mandate for a package
of that PDP's estimated area -- exactly as specified: "Uses Layer 0's
barcode-calibration to convert detected text bounding-box pixel height ->
real-world mm -> compares against LMPC's minimum font-size mandate."

If no barcode/manual calibration is available, this returns UNVERIFIABLE
rather than guessing -- a false PASS here would be worse than no
measurement, and would undermine the tool's use as enforcement evidence.
"""

from typing import Optional
from app.pipeline.types import Calibration, MIN_FONT_SIZE_MM_BY_AREA


def estimate_pdp_area_cm2(image_width_px: float, image_height_px: float, calibration: Calibration) -> Optional[float]:
    if not calibration.found or not calibration.px_per_mm:
        return None
    width_mm = image_width_px / calibration.px_per_mm
    height_mm = image_height_px / calibration.px_per_mm
    return (width_mm / 10) * (height_mm / 10)


def min_required_mm_for_area(area_cm2: float) -> float:
    for max_area, min_mm in MIN_FONT_SIZE_MM_BY_AREA:
        if area_cm2 <= max_area:
            return min_mm
    return MIN_FONT_SIZE_MM_BY_AREA[-1][1]


def check_font_size(
    font_height_px: Optional[float],
    image_width_px: float,
    image_height_px: float,
    calibration: Calibration,
):
    if font_height_px is None:
        return {"status": "UNVERIFIABLE", "font_height_mm": None, "min_required_mm": None,
                "note": "Field not detected; font size cannot be assessed."}

    if not calibration.found or not calibration.px_per_mm:
        return {
            "status": "UNVERIFIABLE",
            "font_height_mm": None,
            "min_required_mm": None,
            "note": (
                "No barcode detected in frame for calibration. Font height cannot be converted to "
                "millimetres — retake the photo with the barcode visible, or use manual calibration."
            ),
        }

    font_height_mm = font_height_px / calibration.px_per_mm
    area_cm2 = estimate_pdp_area_cm2(image_width_px, image_height_px, calibration)
    min_required_mm = min_required_mm_for_area(area_cm2) if area_cm2 is not None else 1.0

    status = "PASS" if font_height_mm >= min_required_mm else "FAIL"
    note = (
        f"Measured {font_height_mm:.2f}mm >= required {min_required_mm:.1f}mm."
        if status == "PASS"
        else f"Measured {font_height_mm:.2f}mm is below the required {min_required_mm:.1f}mm minimum for a PDP of this size."
    )

    return {"status": status, "font_height_mm": font_height_mm, "min_required_mm": min_required_mm, "note": note}
