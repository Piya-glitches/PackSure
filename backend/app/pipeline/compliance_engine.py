"""
ORCHESTRATOR -- Layer 0-4 pipeline, wired to the systechchange.pdf stack:

  raw photo
    -> [L0] zxing-cpp barcode calibration on the RAW frame
    -> [L1] YOLOv8n-seg PDP mask (contour fallback) -> homography OR cylindrical unwarp
    -> [L0] barcode scale converted into the dewarped pixel space
    -> [L0] quality gate on the dewarped label
    -> [L2] PaddleOCR on CLAHE image   ||  [L3] Qwen2.5-VL on the natural-contrast image (parallel)
    -> [L3] VLM-primary / rule-fallback field merge
    -> [L4] format validators + font-size-in-mm + responsible-party resolution

SCALE CONVERSION (important): the barcode is measured in RAW pixels, but OCR
boxes live in DEWARPED pixels (1200px wide crop). px/mm measured on the raw
frame is therefore wrong for the font-size check unless it is carried through
the same geometric transform. _OutputMapper does that (local Jacobian scale for
homography; arcsin-aware scale for the cylindrical path) and also maps the
barcode bbox so the frontend overlay lands in the right place.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Dict, Optional

import cv2
import numpy as np

from app.pipeline.types import (
    Calibration, BBox, ComplianceReport, FieldValidation, Violation, PipelineStageLog, FIELD_WEIGHTS,
)
from app.pipeline.quality_gate import run_quality_gate
from app.pipeline.barcode_calibration import detect_barcode_calibration
from app.pipeline.pdp_segmentation import detect_pdp_mask
from app.pipeline.dewarp import select_dewarp, apply_clahe, estimate_cylinder_radius_px
from app.pipeline.text_detection_ocr import run_ocr, get_reader
from app.pipeline.field_classifier import classify_fields, fetch_vlm_fields, vlm_enabled
from app.pipeline.font_size import check_font_size
from app.pipeline.responsible_party import resolve_responsible_party
from app.pipeline.rule_validators import validate_field_format

MAX_INPUT_SIDE = 3200          # cap 12MP+ phone photos; barcode stays well above 2px/module
_OCR_LOCK = threading.Lock()   # Paddle predictors are not thread-safe; FastAPI runs sync routes in a threadpool


# ---------------------------------------------------------------------------
# Raw-frame -> dewarped-frame mapping
# ---------------------------------------------------------------------------

class _OutputMapper:
    def __init__(self, method: str, quad, mask: Optional[np.ndarray], out_shape):
        self.method = method
        self.out_h, self.out_w = out_shape[:2]
        if method == "cylindrical":
            ys, xs = np.where(mask > 0)
            self.x0, self.y0 = int(xs.min()), int(ys.min())
            self.cw, self.ch = int(xs.max()) - self.x0, int(ys.max()) - self.y0
            radius = estimate_cylinder_radius_px(mask)
            self.r = float(min(radius, self.cw)) if radius else float(self.cw)
            if self.cw <= 0 or self.ch <= 0:
                raise ValueError("degenerate mask")
        else:
            src = np.array(quad, dtype=np.float32)
            dst = np.array([[0, 0], [self.out_w, 0], [self.out_w, self.out_h], [0, self.out_h]], dtype=np.float32)
            self.H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
            self.avg_scale = self.out_w / max(
                float(np.mean([np.hypot(*(np.array(quad[1]) - quad[0])), np.hypot(*(np.array(quad[2]) - quad[3]))])), 1.0)

    def points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.method == "cylindrical":
            xc = self.cw / 2.0
            t = np.clip((pts[:, 0] - self.x0 - xc) / self.r, -1.0, 1.0)
            u = (self.r * np.arcsin(t) + xc) * (self.out_w - 1) / self.cw
            v = (pts[:, 1] - self.y0) * (self.out_h - 1) / self.ch
            return np.stack([u, v], axis=1)
        return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.H).reshape(-1, 2)

    def scale_at(self, x: float, y: float) -> float:
        """Local linear scale (output px per raw px)."""
        if self.method == "cylindrical":
            t = np.clip((x - self.x0 - self.cw / 2.0) / self.r, -0.98, 0.98)
            sx = (self.out_w - 1) / self.cw / np.sqrt(1.0 - t * t)
            sy = (self.out_h - 1) / self.ch
            return float(np.sqrt(sx * sy))
        w = self.H[2, 0] * x + self.H[2, 1] * y + self.H[2, 2]
        if w <= 1e-9:
            return float(self.avg_scale)
        return float(np.sqrt(abs(np.linalg.det(self.H)) / (w ** 3)))     # sqrt(|det J|), det J = det(H)/w^3


def _calibration_to_output_space(calib: Calibration, mapper: _OutputMapper) -> Calibration:
    b = calib.bbox
    cx, cy = b.x + b.w / 2.0, b.y + b.h / 2.0
    s = mapper.scale_at(cx, cy)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("bad scale")
    corners = np.array([[b.x, b.y], [b.x + b.w, b.y], [b.x + b.w, b.y + b.h], [b.x, b.y + b.h]])
    m = mapper.points(corners)
    x0, y0, x1, y1 = m[:, 0].min(), m[:, 1].min(), m[:, 0].max(), m[:, 1].max()
    return replace(
        calib,
        pixel_width=calib.pixel_width * s,
        px_per_mm=calib.px_per_mm * s,
        bbox=BBox(x0, y0, x1 - x0, y1 - y0),
    )


def _cap_resolution(image: np.ndarray, manual_quad):
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= MAX_INPUT_SIDE:
        return image, manual_quad
    s = MAX_INPUT_SIDE / longest
    image = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    if manual_quad:
        manual_quad = [(x * s, y * s) for x, y in manual_quad]
    return image, manual_quad


def _safe_vlm(image):
    try:
        return fetch_vlm_fields(image)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(image_bgr: np.ndarray, manual_quad=None):
    log = []

    def timed(label, fn, note="ok"):
        t0 = time.perf_counter()
        result = fn()
        log.append(PipelineStageLog(stage=label, duration_ms=(time.perf_counter() - t0) * 1000, note=note))
        return result

    raw, manual_quad = _cap_resolution(image_bgr, manual_quad)

    # L0: barcode on the RAW frame (never after CLAHE/homography)
    raw_calib = timed("barcode_raw", lambda: detect_barcode_calibration(raw))

    # L1: PDP segmentation + dewarp
    if manual_quad:
        mask, quad, pdp_method = None, manual_quad, "manual"
    else:
        mask, quad, pdp_method = timed("pdp_segmentation", lambda: detect_pdp_mask(raw))
        log[-1].note = f"method={pdp_method}"
    warped, dewarp_method = timed("dewarp", lambda: select_dewarp(raw, quad, mask))
    log[-1].note = f"method={dewarp_method}"

    # L0: bring the barcode scale into dewarped pixel space
    calibration = None
    if raw_calib.found:
        try:
            mapper = _OutputMapper(dewarp_method, quad, mask, warped.shape)
            calibration = _calibration_to_output_space(raw_calib, mapper)
            log.append(PipelineStageLog("calibration_transfer", 0.0,
                                        f"raw {raw_calib.px_per_mm:.2f} -> output {calibration.px_per_mm:.2f} px/mm"))
        except Exception as exc:
            log.append(PipelineStageLog("calibration_transfer", 0.0, f"failed ({exc}); retrying on dewarped image"))
    if calibration is None:
        # Barcode outside/again undecodable on raw: try the dewarped, pre-CLAHE image
        # (already in output pixel space, so no conversion needed).
        calibration = timed("barcode_warped_fallback", lambda: detect_barcode_calibration(warped))

    enhanced = apply_clahe(warped)

    # Blur/brightness on the dewarped label BEFORE CLAHE (CLAHE inflates Laplacian variance).
    quality = timed("quality_gate", lambda: run_quality_gate(warped))

    # L2 (PaddleOCR, CLAHE image) in parallel with L3 VLM (natural-contrast image)
    with ThreadPoolExecutor(max_workers=1) as pool:
        vlm_future = pool.submit(_safe_vlm, warped)

        def _ocr():
            with _OCR_LOCK:
                return run_ocr(enhanced)

        raw_text, words = timed("ocr", _ocr)
        vlm_fields: Optional[Dict[str, Optional[str]]] = timed("vlm_wait", lambda: vlm_future.result())
    log[-1].note = "vlm=ok" if vlm_fields is not None else "vlm=unavailable -> rule-based fallback"

    extractions = timed("field_classification", lambda: classify_fields(words, raw_text, vlm_fields=vlm_fields))

    # L4
    fields, violations = [], []
    img_h, img_w = enhanced.shape[:2]

    for extraction in extractions:
        format_result = validate_field_format(extraction)
        font_check = check_font_size(extraction.font_height_px, img_w, img_h, calibration)

        status = format_result["status"]
        if status == "PASS" and font_check["status"] == "FAIL":
            status = "FAIL"
            violations.append(Violation(
                rule_code=f"LMPC_{extraction.field_key}_FONT_SIZE",
                severity="MAJOR",
                description=f"{extraction.field_key}: {font_check['note']}",
                citation="PC Rules 2011, Rule 9 (readability/character size)",
            ))
        if format_result["violation"]:
            v = format_result["violation"]
            violations.append(Violation(v["rule_code"], v["severity"], v["description"], v["citation"]))

        fields.append(FieldValidation(
            field_key=extraction.field_key,
            extracted_text=extraction.extracted_text,
            confidence=extraction.confidence,
            bbox=extraction.bbox,
            font_height_px=extraction.font_height_px,
            font_height_mm=font_check["font_height_mm"],
            min_required_mm=font_check["min_required_mm"],
            status=status,
            notes=font_check["note"] if (status == "FAIL" and font_check["status"] == "FAIL"
                                         and format_result["status"] == "PASS") else format_result["notes"],
        ))

    party_text = raw_text
    if vlm_fields and vlm_fields.get("MFR_ADDRESS"):
        party_text = f"{raw_text}\n{vlm_fields['MFR_ADDRESS']}"
    responsible_party = timed("responsible_party", lambda: resolve_responsible_party(party_text))
    if not responsible_party.is_legally_responsible:
        violations.append(Violation(
            rule_code="LMPC_RESPONSIBLE_PARTY",
            severity="CRITICAL" if responsible_party.role == "NOT_FOUND" else "MAJOR",
            description=responsible_party.reasoning,
            citation="PC Rules 2011, Rule 6(1)(a) & Rule 27 (responsibility)",
        ))

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

    critical = sum(1 for v in violations if v.severity == "CRITICAL")
    major = sum(1 for v in violations if v.severity == "MAJOR")

    if not quality.passed:
        overall_status = "NEEDS_REVIEW"
    elif critical > 0:
        overall_status = "NON_COMPLIANT"
    elif major > 0:
        overall_status = "NEEDS_REVIEW" if compliance_score >= 70 else "NON_COMPLIANT"
    elif not calibration.found:
        overall_status = "NEEDS_REVIEW"     # font-size checks were unverifiable: don't stamp COMPLIANT
    else:
        overall_status = "COMPLIANT"

    report = ComplianceReport(
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
    )
    return report, enhanced


# ---------------------------------------------------------------------------
# Startup warm-up (called from FastAPI lifespan) and helpers
# ---------------------------------------------------------------------------

MODEL_STATUS: Dict[str, str] = {}


def warm_up() -> Dict[str, str]:
    """Loads every model once so the first real request isn't a cold start.
    Each component is independent and non-fatal."""
    try:
        get_reader()
        MODEL_STATUS["ocr"] = "paddleocr ready"
    except Exception as exc:
        MODEL_STATUS["ocr"] = f"FAILED: {exc}"
    try:
        from app.pipeline.pdp_segmentation import _try_load_seg_model
        MODEL_STATUS["pdp"] = "yolov8n-seg loaded" if _try_load_seg_model() is not None else "contour_fallback (no checkpoint in models/)"
    except Exception as exc:
        MODEL_STATUS["pdp"] = f"FAILED: {exc}"
    try:
        MODEL_STATUS["vlm"] = "ollama reachable" if vlm_enabled() else "unavailable -> rule-based extraction"
    except Exception as exc:
        MODEL_STATUS["vlm"] = f"FAILED: {exc}"
    return dict(MODEL_STATUS)


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
