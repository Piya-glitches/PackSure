from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "public"


class UserOut(BaseModel):
    id: str
    username: str
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class FieldOut(BaseModel):
    field_key: str
    extracted_text: Optional[str]
    confidence: float
    bounding_box_json: Optional[str]
    font_height_px: Optional[float]
    font_height_mm: Optional[float]
    min_required_mm: Optional[float]
    status: str
    notes: Optional[str]

    class Config:
        from_attributes = True


class ViolationOut(BaseModel):
    rule_code: str
    severity: str
    description: str
    citation: str

    class Config:
        from_attributes = True


class ScanSummary(BaseModel):
    id: str
    created_at: datetime
    product_label: str
    overall_status: str
    compliance_score: int
    calibration_source: Optional[str]

    class Config:
        from_attributes = True


class ScanDetail(ScanSummary):
    image_data_url: str
    calibration_factor: Optional[float]
    pdp_detection_method: Optional[str]
    ocr_raw_text: str
    fields: List[FieldOut]
    violations: List[ViolationOut]

    class Config:
        from_attributes = True


class ScanCreateRequest(BaseModel):
    product_label: str
    image_data_url: str  # base64 data URL from the frontend camera/upload
    manual_quad: Optional[List[List[float]]] = None  # optional [[x,y]*4] override for PDP corners


class SaveScanRequest(BaseModel):
    product_label: str
    processed_image_data_url: str
    report: dict
