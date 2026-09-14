export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Calibration {
  found: boolean;
  symbology: string;
  raw_value: string | null;
  px_per_mm: number | null;
  physical_width_mm: number | null;
  bbox: BBox | null;
}

export interface QualityGate {
  passed: boolean;
  laplacian_variance: number;
  brightness_mean: number;
  reasons: string[];
}

export interface FieldResult {
  field_key: string;
  extracted_text: string | null;
  confidence: number;
  bbox: BBox | null;
  font_height_px: number | null;
  font_height_mm: number | null;
  min_required_mm: number | null;
  status: "PASS" | "FAIL" | "WARN" | "NOT_FOUND";
  notes: string;
}

export interface ResponsibleParty {
  role: string;
  matched_phrase: string | null;
  address: string | null;
  is_legally_responsible: boolean;
  reasoning: string;
}

export interface ViolationItem {
  rule_code: string;
  severity: "CRITICAL" | "MAJOR" | "MINOR";
  description: string;
  citation: string;
}

export interface ComplianceReport {
  overall_status: "COMPLIANT" | "NON_COMPLIANT" | "NEEDS_REVIEW";
  compliance_score: number;
  calibration: Calibration;
  quality_gate: QualityGate;
  pdp_detection_method: string;
  fields: FieldResult[];
  responsible_party: ResponsibleParty;
  violations: ViolationItem[];
  ocr_raw_text: string;
  processing_log: { stage: string; duration_ms: number; note: string }[];
}

export const FIELD_LABELS: Record<string, string> = {
  MFR_ADDRESS: "Manufacturer / Packer / Importer Address",
  COMMON_NAME: "Common / Generic Name",
  NET_QTY: "Net Quantity",
  MFG_DATE: "Month & Year of Manufacture/Packing/Import",
  MRP: "Maximum Retail Price (incl. of all taxes)",
  CONSUMER_CARE: "Consumer Care Details",
  COUNTRY_OF_ORIGIN: "Country of Origin",
  UNIT_PRICE: "Unit Sale Price",
};
