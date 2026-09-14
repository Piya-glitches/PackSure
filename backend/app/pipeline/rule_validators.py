"""
LAYER 4 -- DETERMINISTIC RULE VALIDATORS

Per the architecture: "Deterministic rule functions per field (e.g. MRP
regex: currency + amount + 'inclusive of all taxes' phrase check; date
format validators; unit standardization checks g/kg/ml/l) ... Fast,
explainable, auditable — a judge or a real Legal Metrology officer needs
to see why something failed, not a black-box score."

Intentionally not ML-based: format compliance has a deterministic answer,
and a plain function here is more accurate, faster, and fully auditable
than any model would be.
"""

import re
from app.pipeline.types import FieldExtraction, RULE_CITATIONS, FIELD_LABELS


def _fail(field_key, description, severity="MAJOR"):
    return {
        "status": "FAIL",
        "notes": description,
        "violation": {
            "rule_code": f"LMPC_{field_key}_FORMAT",
            "severity": severity,
            "description": f"{FIELD_LABELS[field_key]}: {description}",
            "citation": RULE_CITATIONS[field_key],
        },
    }


def _not_found(field_key):
    return {
        "status": "NOT_FOUND",
        "notes": "Field was not detected anywhere on the scanned label.",
        "violation": {
            "rule_code": f"LMPC_{field_key}_MISSING",
            "severity": "CRITICAL",
            "description": f"{FIELD_LABELS[field_key]} is a mandatory declaration and was not found on the label.",
            "citation": RULE_CITATIONS[field_key],
        },
    }


def _pass(notes):
    return {"status": "PASS", "notes": notes, "violation": None}


def validate_field_format(extraction: FieldExtraction):
    field_key, text, confidence = extraction.field_key, extraction.extracted_text, extraction.confidence

    if not text or confidence < 0.35:
        return _not_found(field_key)

    if field_key == "MRP":
        has_currency = bool(re.search(r"(?:rs\.?|inr|₹)", text, re.I))
        has_amount = bool(re.search(r"\d{1,4}(?:[.,]\d{1,2})?", text))
        has_inclusive = bool(re.search(r"incl(?:usive)?\.?\s*of\s*all\s*tax", text, re.I))
        if not has_currency or not has_amount:
            return _fail(field_key, "MRP amount or currency symbol not clearly recognised.", "CRITICAL")
        if not has_inclusive:
            return _fail(
                field_key,
                'MRP is declared but the mandatory "inclusive of all taxes" qualifier was not detected nearby '
                "— Rule 18 requires MRP to explicitly state it is tax-inclusive.",
                "MAJOR",
            )
        return _pass("MRP declaration with currency symbol and tax-inclusive clause detected.")

    if field_key == "NET_QTY":
        valid_unit = bool(re.search(r"\d+(\.\d+)?\s?(g|gm|gms|kg|ml|l|litre|liter|mg)\b", text, re.I))
        if not valid_unit:
            return _fail(field_key, "Net quantity is declared but not in a standard metric unit (g/kg/ml/l).", "MAJOR")
        return _pass("Net quantity declared in a standard metric unit.")

    if field_key == "MFG_DATE":
        valid_date = bool(
            re.search(
                r"(0[1-9]|1[0-2])[/\-.](\d{2,4})|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s?\d{2,4}",
                text,
                re.I,
            )
        )
        if not valid_date:
            return _fail(field_key, "A manufacturing/packing date phrase was found but no valid month/year value could be parsed.", "MAJOR")
        return _pass("Manufacture/packing date present in a recognisable month-year format.")

    if field_key == "MFR_ADDRESS":
        has_pin = bool(re.search(r"\b\d{6}\b", text))
        if not has_pin:
            return _fail(field_key, "An address-like declaration was found but no 6-digit PIN code was detected.", "MAJOR")
        return _pass("Address declaration includes a PIN code.")

    if field_key == "CONSUMER_CARE":
        has_contact = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b(1800|\+?91)[\d\s-]{6,}", text, re.I))
        if not has_contact:
            return _fail(field_key, "Consumer care section found but no valid phone number or email could be parsed.", "MINOR")
        return _pass("Consumer care contact (phone or email) detected.")

    if field_key == "COMMON_NAME":
        if len(text.strip()) < 3:
            return _fail(field_key, "Common/generic name text is too short/unclear to confirm.", "MINOR")
        return _pass("Common/generic product name detected.")

    return _pass("No specific format rule configured for this field.")
