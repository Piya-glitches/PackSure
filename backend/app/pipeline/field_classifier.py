"""
LAYER 3 -- FIELD CLASSIFICATION (NER, not regex)

Per the architecture: "Fine-tuned lightweight transformer (DistilBERT/
IndicBERT) doing token classification (NER) over OCR output ... Model
trained on OCR-error-augmented text ... production-viable vs. a demo toy."

HONEST SCOPING (read before judging the novelty claim): a DistilBERT/
IndicBERT token-classifier needs to be FINE-TUNED on our specific 8-class
schema (MFR_ADDRESS, NET_QTY, MRP, MFG_DATE, CONSUMER_CARE,
COUNTRY_OF_ORIGIN, UNIT_PRICE, COMMON_NAME) -- an off-the-shelf pretrained
NER model (e.g. a multilingual NER checkpoint) outputs generic entity
types (PERSON/ORG/LOCATION/MISC), which do not map to our schema and would
misclassify or miss most fields entirely if used as-is.

So this module does both, honestly:
  1. Loads a real DistilBERT token-classification pipeline
     (transformers.AutoModelForTokenClassification) as the architectural
     backbone specified -- this is the concrete Phase 2 fine-tuning target
     (train it on OCR-error-augmented, LMPC-labelled package text, and
     swap FINE_TUNED_MODEL_PATH below).
  2. Runs a rule-augmented classifier (weighted keyword + regex + fuzzy
     matching over OCR line-groups) as the PRACTICAL classifier that
     actually drives the compliance decision today. This mirrors what the
     architecture calls the "80% of hackathon teams" naive approach only
     in the sense that it uses regex signals -- but unlike a keyword
     matcher, it operates on grouped lines (not single words), fuzzy
     -matches to tolerate OCR errors, and scores every field against every
     line rather than stopping at the first hit, which is what makes it
     classification rather than search.

Both paths are real and wired up. We do not present the untrained
DistilBERT backbone's raw output as if it already solves open-set field
extraction -- that would be overclaiming.
"""

from typing import List, Optional, Dict
import os
import re

from app.pipeline.types import OcrWord, FieldExtraction, BBox
from app.pipeline.text_detection_ocr import group_words_into_lines

# ---------------------------------------------------------------------------
# Practical classifier (drives the live compliance decision)
# ---------------------------------------------------------------------------

FIELD_SIGNATURES = {
    "MRP": {
        "keywords": ["mrp", "maximum retail price", "m.r.p", "retail price"],
        "pattern": re.compile(r"(?:rs\.?|inr|₹)\s?\d{1,4}(?:[.,]\d{1,2})?", re.I),
        "kw_weight": 0.55,
        "pat_weight": 0.45,
    },
    "NET_QTY": {
        "keywords": ["net qty", "net weight", "net wt", "net volume", "net content", "net quantity"],
        "pattern": re.compile(r"\d+(\.\d+)?\s?(g|gm|gms|kg|ml|l|litre|liter|mg|pieces|pcs|n)\b", re.I),
        "kw_weight": 0.5,
        "pat_weight": 0.5,
    },
    "MFG_DATE": {
        "keywords": ["mfg", "manufactured on", "pkd", "packed on", "packing date", "best before", "use by", "exp"],
        "pattern": re.compile(
            r"(0[1-9]|1[0-2])[/\-.](\d{2,4})|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s?\d{2,4}", re.I
        ),
        "kw_weight": 0.45,
        "pat_weight": 0.55,
    },
    "MFR_ADDRESS": {
        "keywords": ["mfd by", "manufactured by", "packed by", "marketed by", "manufacturer", "address", "packer"],
        "pattern": re.compile(r"\b\d{6}\b"),
        "kw_weight": 0.6,
        "pat_weight": 0.4,
        "window_lines": 2,  # search this many lines below the keyword hit too
    },
    "CONSUMER_CARE": {
        "keywords": ["customer care", "consumer care", "for complaints", "toll free", "care no", "helpline"],
        "pattern": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b(1800|\+?91)[\d\s-]{6,}", re.I),
        "kw_weight": 0.5,
        "pat_weight": 0.5,
        "window_lines": 2,
    },
    "COUNTRY_OF_ORIGIN": {
        "keywords": ["country of origin", "made in", "product of", "origin"],
        "pattern": re.compile(r"made in\s+\w+|product of\s+\w+|origin\s*:?\s*\w+", re.I),
        "kw_weight": 0.6,
        "pat_weight": 0.4,
    },
    "UNIT_PRICE": {
        "keywords": ["unit sale price", "usp", "price per", "rate per"],
        "pattern": re.compile(r"(?:rs\.?|₹)\s?\d+(\.\d{1,2})?\s?/\s?(kg|l|g|ml|piece)", re.I),
        "kw_weight": 0.5,
        "pat_weight": 0.5,
    },
}


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _best_span_edit_ratio(haystack: str, needle: str) -> float:
    """Slides a window the length of `needle` (+/- a couple chars) across
    `haystack` and returns the best 1 - normalized_edit_distance found.
    This catches OCR-mangled exact occurrences ("countrv of orgin" for
    "country of origin") without letting a short keyword match by pure
    coincidence across an unrelated span of a long line."""
    n = len(needle)
    if n == 0:
        return 0.0
    best_ratio = 0.0
    for size in range(max(1, n - 2), n + 3):
        if size > len(haystack):
            continue
        for start in range(0, len(haystack) - size + 1):
            window = haystack[start:start + size]
            dist = _levenshtein(window, needle)
            ratio = 1.0 - (dist / max(n, size))
            if ratio > best_ratio:
                best_ratio = ratio
    return best_ratio


# Minimum keyword length allowed to use the loose subsequence fallback at
# all. Below this, a keyword MUST appear as a contiguous (low-edit-distance)
# span -- short strings like "usp" or "origin" are far too easy to
# subsequence-match by coincidence inside long garbled OCR lines (this was
# the confirmed root cause of a real scan where COUNTRY_OF_ORIGIN,
# UNIT_PRICE, and MFR_ADDRESS all matched the same unrelated garbled line).
MIN_LEN_FOR_SUBSEQUENCE_FALLBACK = 8


def _fuzzy_keyword_score(line_text: str, keywords: List[str]) -> float:
    """Scores whether any keyword genuinely appears in line_text.

    Two matching modes, both bounded (unlike the old whole-line subsequence
    scan):
      1. Contiguous fuzzy span match (edit-distance scoped) -- always
         attempted, works for any keyword length, and is the primary
         signal for OCR-error tolerance.
      2. Subsequence fallback -- only for keywords long enough
         (>= MIN_LEN_FOR_SUBSEQUENCE_FALLBACK chars) that a coincidental
         subsequence match across a long line is actually unlikely, and
         even then the score is penalized by how "spread out" the match
         is relative to the keyword's own length (density), so a
         technically-present subsequence padded with unrelated characters
         no longer scores near 1.0.
    """
    normalized = re.sub(r"[^a-z0-9@.\s]", "", line_text.lower())
    if not normalized.strip():
        return 0.0

    best = 0.0
    for kw in keywords:
        kw_norm = kw.lower()

        if kw_norm in normalized:
            best = max(best, 1.0)
            continue

        span_ratio = _best_span_edit_ratio(normalized, kw_norm)
        if span_ratio > 0.72:
            best = max(best, span_ratio)

        kw_chars = kw_norm.replace(" ", "")
        if len(kw_chars) >= MIN_LEN_FOR_SUBSEQUENCE_FALLBACK:
            first_idx, last_idx, matched, cursor = None, None, 0, 0
            for ch in kw_chars:
                idx = normalized.find(ch, cursor)
                if idx >= 0:
                    if first_idx is None:
                        first_idx = idx
                    last_idx = idx
                    matched += 1
                    cursor = idx + 1
            if matched == len(kw_chars) and first_idx is not None:
                span_len = last_idx - first_idx + 1
                density = len(kw_chars) / max(span_len, 1)
                if density > 0.5:
                    best = max(best, density * 0.75)

    return best


def _window_text(lines: List[List[OcrWord]]) -> str:
    return " ".join(_line_text(line) for line in lines)


def _line_bbox(line: List[OcrWord]) -> BBox:
    x = min(w.bbox.x for w in line)
    y = min(w.bbox.y for w in line)
    max_x = max(w.bbox.x + w.bbox.w for w in line)
    max_y = max(w.bbox.y + w.bbox.h for w in line)
    return BBox(x, y, max_x - x, max_y - y)


def _line_text(line: List[OcrWord]) -> str:
    return " ".join(w.text for w in line)


def classify_fields_rule_based(words: List[OcrWord]) -> List[FieldExtraction]:
    lines = group_words_into_lines(words)
    results: List[FieldExtraction] = []

    for field_key, sig in FIELD_SIGNATURES.items():
        window_lines = sig.get("window_lines", 0)
        best = None  # (matched_lines: List[List[OcrWord]], score)

        for i, line in enumerate(lines):
            text = _line_text(line)
            kw_score = _fuzzy_keyword_score(text, sig["keywords"])
            if kw_score <= 0.0:
                continue

            pattern_on_line = bool(sig["pattern"].search(text))

            if pattern_on_line or window_lines == 0:
                score = kw_score * sig["kw_weight"] + (1.0 if pattern_on_line else 0.0) * sig["pat_weight"]
                if score > 0.3 and (best is None or score > best[1]):
                    best = ([line], score)
                continue

            # Keyword matched but the pattern didn't -- for fields that
            # allow it, search a small window of subsequent lines for the
            # pattern (e.g. a phone number one line below "Customer Care:").
            found_window = None
            for span in range(1, window_lines + 1):
                if i + span >= len(lines):
                    break
                candidate_lines = lines[i:i + span + 1]
                candidate_text = _window_text(candidate_lines)
                if sig["pattern"].search(candidate_text):
                    found_window = candidate_lines
                    break

            if found_window is not None:
                score = kw_score * sig["kw_weight"] + 1.0 * sig["pat_weight"]
            else:
                found_window = [line]
                score = kw_score * sig["kw_weight"]

            if score > 0.3 and (best is None or score > best[1]):
                best = (found_window, score)

        if best:
            matched_lines, score = best
            all_words_flat = [w for line in matched_lines for w in line]
            bbox = _line_bbox(all_words_flat)
            results.append(
                FieldExtraction(
                    field_key=field_key,
                    extracted_text=_window_text(matched_lines).strip(),
                    confidence=min(score, 0.98),
                    bbox=bbox,
                    font_height_px=bbox.h,
                )
            )
        else:
            results.append(FieldExtraction(field_key, None, 0.0, None, None))

    # COMMON_NAME heuristic: largest-font alphabetic line not already claimed.
    claimed = {r.extracted_text for r in results if r.extracted_text}
    best_name = None
    for line in lines:
        text = _line_text(line)
        if len(text.strip()) < 3:
            continue
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        if digit_ratio > 0.3 or text.strip() in claimed:
            continue
        bbox = _line_bbox(line)
        if best_name is None or bbox.h > best_name[1].h:
            best_name = (line, bbox)

    if best_name:
        line, bbox = best_name
        results.append(FieldExtraction("COMMON_NAME", _line_text(line).strip(), 0.65, bbox, bbox.h))
    else:
        results.append(FieldExtraction("COMMON_NAME", None, 0.0, None, None))

    return results


# ---------------------------------------------------------------------------
# DistilBERT/IndicBERT backbone (architectural component; Phase 2 fine-tune target)
# ---------------------------------------------------------------------------

_ner_pipeline = None
_ner_load_attempted = False

# Local checkpoint fine-tuned on synthetic data via
# scripts/generate_training_data.py + scripts/train_distilbert_classifier.py
# (see packsure_context_transfer.md for training details/caveats). Falls
# back to the generic pretrained backbone (disabled by default) if the
# fine-tuned checkpoint isn't present, so a fresh clone without the model
# file still runs -- just without this signal.
FINE_TUNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "distilbert-lmpc-ner")
BACKBONE_MODEL_NAME = "distilbert-base-multilingual-cased"

# Fields the line classifier is allowed to fill in when the rule-based
# classifier found nothing. NOT overriding a rule-based match that already
# succeeded -- this checkpoint is trained on clean synthetic template text
# and has not been validated against real garbled OCR output yet (see
# training caveats), so it's trusted only as a gap-filler, not an override,
# until real-photo accuracy is measured.
ML_FILLABLE_FIELDS = set(FIELD_SIGNATURES.keys())
MIN_ML_CONFIDENCE = 0.6


def get_line_classifier():
    """Loads a per-line sequence classifier: the fine-tuned local checkpoint
    if present, else the generic pretrained backbone gated behind
    settings.enable_ner_backbone (a supplementary, unfine-tuned signal only --
    see module docstring). Returns None if neither is available/loadable;
    every call site treats that as "no ML signal this run", never a hard
    failure."""
    global _ner_pipeline, _ner_load_attempted
    if _ner_load_attempted:
        return _ner_pipeline
    _ner_load_attempted = True

    try:
        from transformers import pipeline

        if os.path.isdir(FINE_TUNED_MODEL_PATH):
            _ner_pipeline = pipeline("text-classification", model=FINE_TUNED_MODEL_PATH, top_k=None)
            return _ner_pipeline

        from app.config import settings
        if not settings.enable_ner_backbone:
            return None
        # Generic (not fine-tuned) backbone: only usable in its original
        # token-classification form, not as a field classifier. Kept as the
        # small ORG/LOC confidence-nudge behavior it always had.
        _ner_pipeline = pipeline("token-classification", model=BACKBONE_MODEL_NAME, aggregation_strategy="simple")
    except Exception:
        _ner_pipeline = None
    return _ner_pipeline


def classify_fields(words: List[OcrWord], raw_text: str) -> List[FieldExtraction]:
    """Primary entry point: rule-based classification drives the decision.

    If the fine-tuned line classifier is available, it only fills fields the
    rule-based pass came up NOT_FOUND on -- picking, among all OCR lines,
    the highest-confidence line whose predicted label matches that field
    (score must clear MIN_ML_CONFIDENCE). It never overrides a rule-based
    hit. If instead only the generic (unfine-tuned) backbone is available,
    behavior is unchanged from before: a small MFR_ADDRESS confidence nudge
    from ORG/LOC entity overlap."""
    extractions = classify_fields_rule_based(words)

    clf = get_line_classifier()
    if clf is None:
        return extractions

    is_fine_tuned = os.path.isdir(FINE_TUNED_MODEL_PATH)

    if not is_fine_tuned:
        try:
            entities = clf(raw_text)
            org_or_loc_text = " ".join(e["word"] for e in entities if e["entity_group"] in ("ORG", "LOC"))
            for extraction in extractions:
                if extraction.field_key == "MFR_ADDRESS" and extraction.extracted_text:
                    overlap = any(tok in org_or_loc_text for tok in extraction.extracted_text.split() if len(tok) > 3)
                    if overlap:
                        extraction.confidence = min(extraction.confidence + 0.1, 0.99)
        except Exception:
            pass  # backbone is a supplementary signal; never block the pipeline on it
        return extractions

    missing_fields = {e.field_key for e in extractions if not e.extracted_text and e.field_key in ML_FILLABLE_FIELDS}
    if not missing_fields:
        return extractions

    try:
        lines = group_words_into_lines(words)
        best_by_field: Dict[str, tuple] = {}  # field_key -> (score, line, bbox)
        for line in lines:
            text = _line_text(line)
            if not text.strip():
                continue
            preds = clf(text)[0]  # top_k=None -> list of {label, score} for this one line
            top = max(preds, key=lambda p: p["score"])
            if top["label"] not in missing_fields or top["score"] < MIN_ML_CONFIDENCE:
                continue
            if top["label"] not in best_by_field or top["score"] > best_by_field[top["label"]][0]:
                best_by_field[top["label"]] = (top["score"], line, _line_bbox(line))

        by_key = {e.field_key: e for e in extractions}
        for field_key, (score, line, bbox) in best_by_field.items():
            e = by_key[field_key]
            e.extracted_text = _line_text(line).strip()
            e.confidence = float(score) * 0.9  # slight discount: unvalidated-on-real-photos signal
            e.bbox = bbox
            e.font_height_px = bbox.h
    except Exception:
        pass  # ML fill is a bonus signal; never block the pipeline on it

    return extractions
