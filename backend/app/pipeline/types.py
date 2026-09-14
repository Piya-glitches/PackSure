from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

FIELD_LABELS: Dict[str, str] = {
    "MFR_ADDRESS": "Manufacturer / Packer / Importer Address",
    "COMMON_NAME": "Common / Generic Name",
    "NET_QTY": "Net Quantity",
    "MFG_DATE": "Month & Year of Manufacture/Packing/Import",
    "MRP": "Maximum Retail Price (incl. of all taxes)",
    "CONSUMER_CARE": "Consumer Care Details",
    "COUNTRY_OF_ORIGIN": "Country of Origin",
    "UNIT_PRICE": "Unit Sale Price",
}

RULE_CITATIONS: Dict[str, str] = {
    "MFR_ADDRESS": "PC Rules 2011, Rule 6(1)(a)",
    "COMMON_NAME": "PC Rules 2011, Rule 6(1)(b)",
    "NET_QTY": "PC Rules 2011, Rule 6(1)(c) & Rule 8 (std. units)",
    "MFG_DATE": "PC Rules 2011, Rule 6(1)(e)",
    "MRP": "PC Rules 2011, Rule 6(1)(f) & Rule 18 (MRP format)",
    "CONSUMER_CARE": "PC Rules 2011, Rule 6(1)(g)",
    "COUNTRY_OF_ORIGIN": "Legal Metrology Act, 2009 read with Consumer Protection (E-Commerce) Rules",
    "UNIT_PRICE": "PC Rules 2011, Rule 6(1)(f), explanation on multi-piece packages",
}

# LMPC minimum PDP font-size mandate tiers, by principal display panel area.
MIN_FONT_SIZE_MM_BY_AREA: List[Tuple[float, float]] = [
    (100.0, 1.0),
    (500.0, 2.0),
    (float("inf"), 4.0),
]

FIELD_WEIGHTS: Dict[str, int] = {
    "MRP": 18,
    "NET_QTY": 15,
    "MFR_ADDRESS": 15,
    "MFG_DATE": 12,
    "CONSUMER_CARE": 10,
    "COMMON_NAME": 10,
    "COUNTRY_OF_ORIGIN": 10,
    "UNIT_PRICE": 10,
}


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self):
        # EasyOCR/OpenCV/pyzbar frequently return numpy scalar types
        # (numpy.int64, numpy.float64) rather than native Python numbers.
        # FastAPI's jsonable_encoder cannot serialize those directly
        # (raises "numpy.int64 object is not iterable"). Casting here, at
        # the single construction point every bbox flows through, means
        # no downstream code has to remember to do it.
        self.x = float(self.x)
        self.y = float(self.y)
        self.w = float(self.w)
        self.h = float(self.h)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class OcrWord:
    text: str
    confidence: float  # 0-1
    bbox: BBox

    def __post_init__(self):
        self.confidence = float(self.confidence)


@dataclass
class Calibration:
    found: bool
    symbology: str = "NONE"  # EAN_13 | UPC_A | EAN_8 | MANUAL | NONE
    raw_value: Optional[str] = None
    pixel_width: Optional[float] = None
    physical_width_mm: Optional[float] = None
    px_per_mm: Optional[float] = None
    bbox: Optional[BBox] = None

    def __post_init__(self):
        # Same numpy-scalar defensiveness as BBox/OcrWord above.
        if self.pixel_width is not None:
            self.pixel_width = float(self.pixel_width)
        if self.physical_width_mm is not None:
            self.physical_width_mm = float(self.physical_width_mm)
        if self.px_per_mm is not None:
            self.px_per_mm = float(self.px_per_mm)


@dataclass
class QualityGateResult:
    passed: bool
    laplacian_variance: float
    brightness_mean: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class FieldExtraction:
    field_key: str
    extracted_text: Optional[str]
    confidence: float
    bbox: Optional[BBox]
    font_height_px: Optional[float]


@dataclass
class FieldValidation:
    field_key: str
    extracted_text: Optional[str]
    confidence: float
    bbox: Optional[BBox]
    font_height_px: Optional[float]
    font_height_mm: Optional[float]
    min_required_mm: Optional[float]
    status: str  # PASS | FAIL | WARN | NOT_FOUND
    notes: str


@dataclass
class Violation:
    rule_code: str
    severity: str  # CRITICAL | MAJOR | MINOR
    description: str
    citation: str


@dataclass
class ResponsiblePartyResult:
    role: str  # MANUFACTURER | PACKER | IMPORTER | MARKETER | AMBIGUOUS | NOT_FOUND
    matched_phrase: Optional[str]
    address: Optional[str]
    is_legally_responsible: bool
    reasoning: str


@dataclass
class PipelineStageLog:
    stage: str
    duration_ms: float
    note: str


@dataclass
class ComplianceReport:
    overall_status: str
    compliance_score: int
    calibration: Calibration
    quality_gate: QualityGateResult
    pdp_detection_method: str
    fields: List[FieldValidation]
    responsible_party: ResponsiblePartyResult
    violations: List[Violation]
    ocr_raw_text: str
    processing_log: List[PipelineStageLog]
