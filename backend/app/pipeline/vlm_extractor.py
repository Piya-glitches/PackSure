"""
LAYER 3 -- FIELD EXTRACTION, APPROACH A (LOCAL MICRO-VLM)
per systechchange.pdf: "Serve Qwen2.5-VL-3B-Instruct ... Prompt with the
cropped PDP image and an explicit JSON schema for the 8 LMPC mandatory
fields. A VLM natively reads non-standard spatial layouts, handles
rotated dates, and ignores decorative brand slogans."

WHY APPROACH A OVER APPROACH B (LayoutLMv3) FOR THIS PROJECT SPECIFICALLY:
LayoutLMv3 fine-tuning needs ~1,000 real-photo labels with word-level
bounding boxes and BIO tags -- exactly the "labelled dataset we don't
have yet" gap that already blocked the YOLOv8 PDP detector and the
DistilBERT NER model in v1. A pretrained-and-frozen VLM needs zero
task-specific training data: the JSON schema in the prompt IS the
supervision. Given the team's actual constraints (no GPU laptop, days
not weeks, hackathon judged on working behavior), this is the honest
choice, not the easy one -- LayoutLMv3 quality would silently regress to
"can't run it at all" without an annotated dataset this team doesn't have.

RUNTIME: talks to a local Ollama server (http://localhost:11434) running
`ollama pull qwen2.5vl:3b`. This keeps inference fully offline/local like
every other model in this pipeline (EasyOCR/PaddleOCR/YOLO weights are
also all local) rather than adding a paid API dependency. If Ollama isn't
reachable, extract_fields_vlm() returns None and the caller
(field_classifier.py) falls back to the rule-based classifier alone --
same "never block the pipeline on an optional signal" pattern used
throughout this codebase for the DistilBERT backbone and the YOLO PDP
checkpoint.

This module produces the PRIMARY signal when available (unlike v1's
DistilBERT gap-filler, which only filled NOT_FOUND fields) because a VLM
reading the actual image is strictly more informed than a classifier
reading only OCR's already-lossy text output. The rule-based classifier
in field_classifier.py becomes the fallback/cross-check layer, not the
other way around. Every VLM-sourced field is still passed through
rule_validators.py's deterministic format checks --  the VLM's job is
extraction, not judging Rule 18/Rule 9 compliance.
"""

import base64
import json
import os
import re
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests

from app.pipeline.types import FieldExtraction

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VLM_MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "qwen2.5vl:3b")
VLM_TIMEOUT_SECONDS = 45

# Keys MUST stay in sync with field_classifier.FIELD_SIGNATURES /
# types.FIELD_LABELS / types.RULE_CITATIONS.
FIELD_SCHEMA_DESCRIPTION = {
    "MRP": "Maximum Retail Price, including currency symbol and the 'inclusive of all taxes' qualifier if present, exactly as printed.",
    "NET_QTY": "Net quantity/weight/volume declaration with its unit, exactly as printed (e.g. '200 ml', '150 g').",
    "MFG_DATE": "Month and year of manufacture, packing, or import, exactly as printed (e.g. '03/2026', 'Mar 2026').",
    "MFR_ADDRESS": "The full manufacturer/packer/importer name and postal address block, including PIN code if present.",
    "CONSUMER_CARE": "Consumer care phone number and/or email address for complaints.",
    "COUNTRY_OF_ORIGIN": "Country of origin declaration, exactly as printed.",
    "UNIT_PRICE": "Unit sale price (price per standard unit, e.g. 'Rs 99.50 / 100ml'), if printed.",
    "COMMON_NAME": "The common/generic name of the product (not the brand name).",
}

_PROMPT_TEMPLATE = """You are reading a photograph of an Indian packaged-commodity label (Principal Display Panel). Extract ONLY the following fields if they are visibly printed on the label. For each field, return the exact text as printed -- do not paraphrase, translate, or normalize units.

Fields to extract:
{field_list}

Return ONLY a single JSON object, no markdown fences, no commentary, with exactly these keys. If a field is not visible on the label, use null for that key.

{{"MRP": ..., "NET_QTY": ..., "MFG_DATE": ..., "MFR_ADDRESS": ..., "CONSUMER_CARE": ..., "COUNTRY_OF_ORIGIN": ..., "UNIT_PRICE": ..., "COMMON_NAME": ...}}
"""


def _build_prompt() -> str:
    field_list = "\n".join(f"- {k}: {v}" for k, v in FIELD_SCHEMA_DESCRIPTION.items())
    return _PROMPT_TEMPLATE.format(field_list=field_list)


def _encode_image(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Failed to encode image for VLM request.")
    return base64.b64encode(buf.tobytes()).decode()


def _extract_json_object(raw: str) -> Optional[dict]:
    """VLMs frequently wrap JSON in ```json fences or add a leading
    sentence despite instructions -- strip defensively rather than
    trusting the model followed formatting instructions verbatim."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def is_vlm_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def extract_fields_vlm(image_bgr: np.ndarray) -> Optional[Dict[str, Optional[str]]]:
    """Returns {field_key: extracted_text_or_None} on success, or None if
    the VLM is unreachable / returns unparseable output. Never raises --
    callers treat None as 'no VLM signal this run', identical to how
    get_line_classifier() returning None is handled in v1."""
    try:
        image_b64 = _encode_image(image_bgr)
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": VLM_MODEL_NAME,
                "prompt": _build_prompt(),
                "images": [image_b64],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            },
            timeout=VLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        raw_output = response.json().get("response", "")
        parsed = _extract_json_object(raw_output)
        if parsed is None:
            return None

        result: Dict[str, Optional[str]] = {}
        for key in FIELD_SCHEMA_DESCRIPTION:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip() and value.strip().lower() not in ("null", "none", "n/a"):
                result[key] = value.strip()
            else:
                result[key] = None
        return result
    except (requests.RequestException, ValueError, KeyError):
        return None


def vlm_result_to_extractions(vlm_fields: Dict[str, Optional[str]]) -> List[FieldExtraction]:
    """Converts the VLM's flat text answers into FieldExtraction objects.
    bbox/font_height_px are always None here -- a VLM gives you the text,
    not a pixel-accurate bounding box, so font-size-in-mm validation
    (font_size.py) will correctly report UNVERIFIABLE for VLM-sourced
    fields unless the rule-based OCR-line pass also found the same text
    (see field_classifier.py's merge logic, which prefers the OCR-line
    bbox when both signals agree on the text)."""
    extractions = []
    for field_key, text in vlm_fields.items():
        if text:
            extractions.append(FieldExtraction(field_key=field_key, extracted_text=text, confidence=0.85, bbox=None, font_height_px=None))
        else:
            extractions.append(FieldExtraction(field_key=field_key, extracted_text=None, confidence=0.0, bbox=None, font_height_px=None))
    return extractions
