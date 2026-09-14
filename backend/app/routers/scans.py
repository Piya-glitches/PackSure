import base64
import json
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Scan, ComplianceField, Violation, User, Role
from app.schemas import ScanCreateRequest, ScanSummary, ScanDetail, SaveScanRequest
from app.security import get_current_user, require_role
from app.pipeline.compliance_engine import run_full_pipeline, decode_image_bytes, encode_image_to_jpeg_bytes
from app.pipeline.sample_labels import generate_sample_label, list_samples

router = APIRouter(prefix="/scans", tags=["scans"])


def _decode_data_url(data_url: str) -> bytes:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


@router.get("/samples")
def get_samples():
    return {"samples": list_samples()}


@router.post("/run")
def run_scan(payload: ScanCreateRequest, user: Optional[User] = Depends(get_current_user), db: Session = Depends(get_db)):
    """Runs the full pipeline and returns the report WITHOUT persisting it.
    The frontend calls /scans (POST) separately to save, after the user
    reviews the result -- mirrors "scan first, save if you want a record"."""
    try:
        image_bgr = decode_image_bytes(_decode_data_url(payload.image_data_url))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    manual_quad = [tuple(p) for p in payload.manual_quad] if payload.manual_quad else None
    report, processed_image = run_full_pipeline(image_bgr, manual_quad=manual_quad)

    processed_jpeg = encode_image_to_jpeg_bytes(processed_image)
    processed_data_url = "data:image/jpeg;base64," + base64.b64encode(processed_jpeg).decode()

    return {
        "report": _report_to_dict(report),
        "processed_image_data_url": processed_data_url,
    }


@router.post("", response_model=ScanDetail)
def save_scan(
    payload: SaveScanRequest,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = payload.report
    scan = Scan(
        owner_id=user.id if user else None,
        product_label=payload.product_label,
        image_data_url=payload.processed_image_data_url,
        overall_status=report["overall_status"],
        compliance_score=report["compliance_score"],
        calibration_factor=report["calibration"].get("px_per_mm"),
        calibration_source=report["calibration"].get("symbology") if report["calibration"].get("found") else None,
        pdp_detection_method=report.get("pdp_detection_method"),
        ocr_raw_text=report["ocr_raw_text"],
        processing_log_json=json.dumps(report.get("processing_log", [])),
    )
    db.add(scan)
    db.flush()

    for f in report["fields"]:
        db.add(ComplianceField(
            scan_id=scan.id,
            field_key=f["field_key"],
            extracted_text=f.get("extracted_text"),
            confidence=f.get("confidence", 0),
            bounding_box_json=json.dumps(f["bbox"]) if f.get("bbox") else None,
            font_height_px=f.get("font_height_px"),
            font_height_mm=f.get("font_height_mm"),
            min_required_mm=f.get("min_required_mm"),
            status=f["status"],
            notes=f.get("notes"),
        ))

    for v in report["violations"]:
        db.add(Violation(
            scan_id=scan.id,
            rule_code=v["rule_code"],
            severity=v["severity"],
            description=v["description"],
            citation=v["citation"],
        ))

    db.commit()
    db.refresh(scan)
    return scan


@router.get("", response_model=List[ScanSummary])
def list_scans(
    status: Optional[str] = Query(None),
    limit: int = 50,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Scan)
    # Public users see only their own scans; officers/admins see all.
    if user is None or user.role == Role.PUBLIC:
        if user:
            query = query.filter(Scan.owner_id == user.id)
        else:
            query = query.filter(Scan.owner_id.is_(None))
    if status:
        query = query.filter(Scan.overall_status == status)
    return query.order_by(Scan.created_at.desc()).limit(limit).all()


@router.get("/{scan_id}", response_model=ScanDetail)
def get_scan(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return scan


@router.delete("/{scan_id}")
def delete_scan(scan_id: str, user: User = Depends(require_role(Role.ADMIN, Role.OFFICER)), db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
    db.delete(scan)
    db.commit()
    return {"ok": True}


@router.get("/samples/{sample_id}/run")
def run_sample(sample_id: str):
    try:
        image_bgr = generate_sample_label(sample_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown sample id.")

    report, processed_image = run_full_pipeline(image_bgr)
    processed_jpeg = encode_image_to_jpeg_bytes(processed_image)
    processed_data_url = "data:image/jpeg;base64," + base64.b64encode(processed_jpeg).decode()

    return {"report": _report_to_dict(report), "processed_image_data_url": processed_data_url}


def _report_to_dict(report) -> dict:
    return {
        "overall_status": report.overall_status,
        "compliance_score": report.compliance_score,
        "calibration": {
            "found": report.calibration.found,
            "symbology": report.calibration.symbology,
            "raw_value": report.calibration.raw_value,
            "px_per_mm": report.calibration.px_per_mm,
            "physical_width_mm": report.calibration.physical_width_mm,
            "bbox": report.calibration.bbox.to_dict() if report.calibration.bbox else None,
        },
        "quality_gate": {
            "passed": report.quality_gate.passed,
            "laplacian_variance": report.quality_gate.laplacian_variance,
            "brightness_mean": report.quality_gate.brightness_mean,
            "reasons": report.quality_gate.reasons,
        },
        "pdp_detection_method": report.pdp_detection_method,
        "fields": [
            {
                "field_key": f.field_key,
                "extracted_text": f.extracted_text,
                "confidence": f.confidence,
                "bbox": f.bbox.to_dict() if f.bbox else None,
                "font_height_px": f.font_height_px,
                "font_height_mm": f.font_height_mm,
                "min_required_mm": f.min_required_mm,
                "status": f.status,
                "notes": f.notes,
            }
            for f in report.fields
        ],
        "responsible_party": {
            "role": report.responsible_party.role,
            "matched_phrase": report.responsible_party.matched_phrase,
            "address": report.responsible_party.address,
            "is_legally_responsible": report.responsible_party.is_legally_responsible,
            "reasoning": report.responsible_party.reasoning,
        },
        "violations": [
            {"rule_code": v.rule_code, "severity": v.severity, "description": v.description, "citation": v.citation}
            for v in report.violations
        ],
        "ocr_raw_text": report.ocr_raw_text,
        "processing_log": [{"stage": s.stage, "duration_ms": s.duration_ms, "note": s.note} for s in report.processing_log],
    }
