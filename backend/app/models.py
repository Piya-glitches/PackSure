import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
import enum

from app.db import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Role(str, enum.Enum):
    PUBLIC = "public"      # consumer — can scan, view own scans
    OFFICER = "officer"    # enforcement official — can view all scans, dashboard
    ADMIN = "admin"        # can delete records, manage users


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(Role), default=Role.PUBLIC, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    scans = relationship("Scan", back_populates="owner")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(String, primary_key=True, default=gen_uuid)
    owner_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    product_label = Column(String, nullable=False)
    image_data_url = Column(Text, nullable=False)

    overall_status = Column(String, nullable=False, index=True)  # COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW
    compliance_score = Column(Integer, nullable=False)

    calibration_factor = Column(Float, nullable=True)   # px per mm
    calibration_source = Column(String, nullable=True)  # EAN_13 | UPC_A | MANUAL | None

    pdp_detection_method = Column(String, nullable=True)  # "yolov8" | "contour_fallback"
    ocr_raw_text = Column(Text, nullable=False)
    ocr_language = Column(String, default="en+hi")
    processing_log_json = Column(Text, nullable=True)

    owner = relationship("User", back_populates="scans")
    fields = relationship("ComplianceField", back_populates="scan", cascade="all, delete-orphan")
    violations = relationship("Violation", back_populates="scan", cascade="all, delete-orphan")


class ComplianceField(Base):
    __tablename__ = "compliance_fields"

    id = Column(String, primary_key=True, default=gen_uuid)
    scan_id = Column(String, ForeignKey("scans.id"), nullable=False)

    field_key = Column(String, nullable=False)
    extracted_text = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    bounding_box_json = Column(Text, nullable=True)
    font_height_px = Column(Float, nullable=True)
    font_height_mm = Column(Float, nullable=True)
    min_required_mm = Column(Float, nullable=True)
    status = Column(String, nullable=False)  # PASS | FAIL | WARN | NOT_FOUND
    notes = Column(Text, nullable=True)

    scan = relationship("Scan", back_populates="fields")


class Violation(Base):
    __tablename__ = "violations"

    id = Column(String, primary_key=True, default=gen_uuid)
    scan_id = Column(String, ForeignKey("scans.id"), nullable=False)

    rule_code = Column(String, nullable=False)
    severity = Column(String, nullable=False)  # CRITICAL | MAJOR | MINOR
    description = Column(Text, nullable=False)
    citation = Column(String, nullable=False)

    scan = relationship("Scan", back_populates="violations")
