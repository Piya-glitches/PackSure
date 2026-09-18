"""
ORCHESTRATOR -- runs the full Layer 0-4 pipeline end to end and produces a
single ComplianceReport, matching the architecture's "End-to-End Flow":

  Photo/Upload
    -> [Layer 0] Quality check + barcode-based calibration
    -> [Layer 1] PDP detection + dewarping
    -> [Layer 2] Multi-script text detection + OCR
    -> [Layer 3] NER field classification
    -> [Layer 4] Rule validation (format + responsible-party + font-size-in-mm)
    -> Output: Pass/Fail per field, with evidence, citation, and confidence
"""

import time
import cv2
import numpy as np

from app.pipeline.types import (
    ComplianceReport, FieldValidation, Violation, PipelineStageLog, FIELD_WEIGHTS,
)
from app.pipeline.quality_gate import run_quality_gate
from app.pipeline.barcode_calibration import detect_barcode_calibration
from app.pipeline.pdp_detection import detect_pdp_quad
from app.pipeline.dewarp import warp_perspective, apply_clahe
from app.pipeline.text_detection_ocr import run_ocr
from app.pipeline.field_classifier import classify_fields
from app.pipeline.font_size import check_font_size
from app.pipeline.responsible_party import resolve_responsible_party
from app.pipeline.rule_validators import validate_field_format


def run_full_pipeline(image_bgr: np.ndarray, manual_quad=None) -> ComplianceReport:
    log = []

    def timed(label, fn):
        t0 = time.perf_counter()
        result = fn()
        log.append(PipelineStageLog(stage=label, duration_ms=(time.perf_counter() - t0) * 1000, note="ok"))
        return result

    # Layer 1a: PDP localization (YOLOv8 or contour fallback)
    if manual_quad:
        quad, pdp_method = manual_quad, "manual"
    else:
        quad, pdp_method = timed("pdp_detection", lambda: detect_pdp_quad(image_bgr))

    # Layer 1b: perspective correction + CLAHE contrast enhancement
    warped = timed("dewarp", lambda: warp_perspective(image_bgr, quad))
    enhanced = timed("clahe", lambda: apply_clahe(warped))

    # Layer 0: quality gate (run on the corrected image, since that's what OCR actually sees)
    quality = timed("quality_gate", lambda: run_quality_gate(enhanced))

    # Layer 0: barcode calibration.
    # Run on the ORIGINAL image first -- CLAHE's local contrast remapping
    # and the perspective homography warp both distort the sharp black/
    # white module transitions pyzbar depends on to decode a barcode, and
    # in practice this caused real, visibly-scannable barcodes to report
    # as "not found" when calibration ran on the enhanced/warped image
    # instead. Only fall back to the warped/enhanced image if nothing is
    # found on the original (e.g. the barcode sits outside the detected
    # PDP quad in the source photo -- shouldn't normally happen, but a
    # safe fallback rather than a regression). This is also more correct
    # architecturally: Layer 0 calibration is meant to measure the barcode
    # as actually photographed, before any Layer 1 geometric processing.
    def _calibrate():
        cal = detect_barcode_calibration(image_bgr)
        if cal.found:
            return cal
        return detect_barcode_calibration(enhanced)

    calibration = timed("barcode_calibration", _calibrate)

    # Layer 2: CRAFT text detection + CRNN recognition (via EasyOCR)
    raw_text, words = timed("ocr", lambda: run_ocr(enhanced))

    # Layer 3: field classification (rule-based + DistilBERT ensemble signal)
    extractions = timed("field_classification", lambda: classify_fields(words, raw_text))

    # Layer 4: rule validation + font-size + responsible-party resolution
    fields = []
    violations = []
    img_h, img_w = enhanced.shape[:2]

    for extraction in extractions:
        format_result = validate_field_format(extraction)
        font_check = check_font_size(extraction.font_height_px, img_w, img_h, calibration)

        status = format_result["status"]
        if status == "PASS" and font_check["status"] == "FAIL":
            status = "FAIL"
            violations.append(
                Violation(
                    rule_code=f"LMPC_{extraction.field_key}_FONT_SIZE",
                    severity="MAJOR",
                    description=f"{extraction.field_key}: {font_check['note']}",
                    citation="PC Rules 2011, Rule 9 (readability/character size)",
                )
            )

        if format_result["violation"]:
            v = format_result["violation"]
            violations.append(Violation(v["rule_code"], v["severity"], v["description"], v["citation"]))

        fields.append(
            FieldValidation(
                field_key=extraction.field_key,
                extracted_text=extraction.extracted_text,
                confidence=extraction.confidence,
                bbox=extraction.bbox,
                font_height_px=extraction.font_height_px,
                font_height_mm=font_check["font_height_mm"],
                min_required_mm=font_check["min_required_mm"],
                status=status,
                notes=font_check["note"] if (status == "FAIL" and font_check["status"] == "FAIL" and format_result["status"] == "PASS") else format_result["notes"],
            )
        )

    responsible_party = timed("responsible_party", lambda: resolve_responsible_party(raw_text))
    if not responsible_party.is_legally_responsible:
        violations.append(
            Violation(
                rule_code="LMPC_RESPONSIBLE_PARTY",
                severity="CRITICAL" if responsible_party.role == "NOT_FOUND" else "MAJOR",
                description=responsible_party.reasoning,
                citation="PC Rules 2011, Rule 6(1)(a) & Rule 27 (responsibility)",
            )
        )

    # Weighted scoring
    earned, total = 0.0, 0.0
    for f in fields:
        weight = FIELD_WEIGHTS[f.field_key]
        total += weight
        if f.status == "PASS":
            earned += weight
        elif f.status == "WARN":
            earned += weight * 0.5
    total += 10
    if responsible_party.is_legally_responsible:
        earned += 10

    compliance_score = round((earned / total) * 100) if total else 0

    critical_count = sum(1 for v in violations if v.severity == "CRITICAL")
    major_count = sum(1 for v in violations if v.severity == "MAJOR")

    if not quality.passed:
        overall_status = "NEEDS_REVIEW"
    elif critical_count > 0:
        overall_status = "NON_COMPLIANT"
    elif major_count > 0:
        overall_status = "NEEDS_REVIEW" if compliance_score >= 70 else "NON_COMPLIANT"
    else:
        overall_status = "COMPLIANT"

    return ComplianceReport(
        overall_status=overall_status,
        compliance_score=compliance_score,
        calibration=calibration,
        quality_gate=quality,
        pdp_detection_method=pdp_method,
        fields=fields,
        responsible_party=responsible_party,
        violations=violations,
        ocr_raw_text=raw_text,
        processing_log=log,
    ), enhanced


def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image data.")
    return image


def encode_image_to_jpeg_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Failed to encode image.")
    return buf.tobytes()
