"""
LAYER 3 -- FIELD EXTRACTION / CLASSIFICATION  (rewritten per systechchange.pdf)

Three signals, in priority order:

  1. VLM (Qwen2.5-VL via Ollama, see vlm_extractor.py)  -- PRIMARY.
     Reads the dewarped label image directly and returns the 8 LMPC fields
     as JSON. Its text is then *grounded* back onto the OCR lines
     (_match_text_to_lines) so every VLM field still gets a pixel bbox and
     a font height for the mm-based readability check.
  2. Rule-based OCR-line classifier                     -- FALLBACK / cross-check.
     Used whole when the VLM is unreachable, and per-field when the VLM
     says "not visible" but the OCR text has a strong keyword+pattern hit.
  3. Fine-tuned DistilBERT line classifier (optional)   -- fills WEAK rule
     fields only. Loads ./models/distilbert-lmpc-ner if present and
     ENABLE_LINE_CLASSIFIER=true. (The old code loaded the *generic*
     pretrained checkpoint, whose classification head is randomly
     initialised -- that was decoration, not a signal, and is removed.)

Bug fixes folded in (systechchange.pdf, Step 1):
  * Keyword matching is token-based, never raw substring. Short keywords
    (mrp, usp, exp, mfg, pkd ...) must match as standalone words.
  * Multi-line fields (CONSUMER_CARE, MFR_ADDRESS) collect the matched line
    plus up to ADJACENT_LINE_WINDOW lines directly below it
    (y_next > y_cur, x-aligned within a tolerance).

Nothing here judges compliance; rule_validators.py does that.
"""

import os
import re
import statistics
import time
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

# vlm_extractor reads OLLAMA_URL / VLM_MODEL_NAME from os.environ at import
# time, and pydantic-settings does NOT export .env into os.environ -- so load
# it here, before the import below.
load_dotenv()

from app.pipeline import vlm_extractor  # noqa: E402
from app.pipeline.types import OcrWord, FieldExtraction, BBox  # noqa: E402
from app.pipeline.text_detection_ocr import group_words_into_lines  # noqa: E402

# Cold CPU inference of a 3B VLM easily exceeds the 45s default in
# vlm_extractor.py; override from the environment without editing that file.
vlm_extractor.VLM_TIMEOUT_SECONDS = int(os.environ.get("VLM_TIMEOUT_SECONDS", "120"))
VLM_MAX_SIDE = int(os.environ.get("VLM_MAX_SIDE", "1280"))

FIELD_ORDER = [
    "MRP", "NET_QTY", "MFG_DATE", "MFR_ADDRESS",
    "CONSUMER_CARE", "COUNTRY_OF_ORIGIN", "UNIT_PRICE", "COMMON_NAME",
]

MULTI_LINE_WINDOW_FIELDS = {"CONSUMER_CARE", "MFR_ADDRESS"}
ADJACENT_LINE_WINDOW = 2
X_ALIGN_TOLERANCE_RATIO = 0.12   # of page width
MAX_VERTICAL_GAP_RATIO = 2.0     # of the previous line's height
RULE_FALLBACK_MIN_CONF = 0.7     # keep a rule hit when the VLM says "null"

PIN_RE = re.compile(r"\b\d{6}\b")
TAX_CLAUSE_RE = re.compile(r"incl(?:usive)?\.?\s*of\s*all\s*tax(?:es)?", re.I)

# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------

FIELD_SIGNATURES = {
    "MRP": {
        "keywords": ["mrp", "maximum retail price", "retail price"],
        "pattern": re.compile(r"(?:rs\.?|inr|₹)\s?\d{1,4}(?:[.,]\d{1,2})?", re.I),
        "kw_weight": 0.55, "pat_weight": 0.45,
    },
    "NET_QTY": {
        "keywords": ["net qty", "net weight", "net wt", "net volume", "net content", "net quantity"],
        "pattern": re.compile(r"\d+(\.\d+)?\s?(g|gm|gms|kg|ml|l|ltr|litre|liter|mg|pcs|pieces|nos?)\b", re.I),
        "kw_weight": 0.5, "pat_weight": 0.5,
    },
    "MFG_DATE": {
        "keywords": ["mfg", "manufactured on", "pkd", "packed on", "packing date", "date of manufacture",
                     "best before", "use by", "exp"],
        "pattern": re.compile(
            r"(0[1-9]|1[0-2])[/\-.](\d{2,4})|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s?\d{2,4}", re.I
        ),
        "kw_weight": 0.45, "pat_weight": 0.55,
    },
    "MFR_ADDRESS": {
        "keywords": ["mfd by", "manufactured by", "packed by", "marketed by", "mktd by", "imported by",
                     "manufacturer", "packer", "address"],
        "pattern": PIN_RE,
        "kw_weight": 0.6, "pat_weight": 0.4,
    },
    "CONSUMER_CARE": {
        "keywords": ["customer care", "consumer care", "contact us", "for complaints", "toll free",
                     "care no", "helpline"],
        "pattern": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b(1800|\+?91)[\d\s-]{6,}", re.I),
        "kw_weight": 0.5, "pat_weight": 0.5,
    },
    "COUNTRY_OF_ORIGIN": {
        "keywords": ["country of origin", "made in", "product of", "origin"],
        "pattern": re.compile(r"made in\s+\w+|product of\s+\w+|origin\s*:?\s*\w+", re.I),
        "kw_weight": 0.6, "pat_weight": 0.4,
    },
    "UNIT_PRICE": {
        "keywords": ["unit sale price", "usp", "price per", "rate per"],
        "pattern": re.compile(r"(?:rs\.?|₹)\s?\d+(?:\.\d{1,2})?\s?(?:/|per)\s?\d*\s?(?:kg|l|g|ml|piece|pc)\b", re.I),
        "kw_weight": 0.5, "pat_weight": 0.5,
    },
}


def _norm(s: str) -> str:
    s = s.lower().replace(".", "")          # "M.R.P." -> "mrp"
    return re.sub(r"[^a-z0-9@\s]", " ", s)


def _tokens(s: str) -> List[str]:
    return _norm(s).split()


def _keyword_score(line_text: str, keywords: List[str]) -> float:
    """Token-based keyword match. 1.0 = exact phrase on word boundaries.
    Short single-word keywords only ever match exactly (no fuzzy, no
    substring), so 'usp' can no longer fire inside OCR noise."""
    line_tokens = _tokens(line_text)
    if not line_tokens:
        return 0.0
    joined = " " + " ".join(line_tokens) + " "
    best = 0.0
    for kw in keywords:
        kw_tokens = _tokens(kw)
        if not kw_tokens:
            continue
        if (" " + " ".join(kw_tokens) + " ") in joined:
            return 1.0
        if len(kw_tokens) == 1 and len(kw_tokens[0]) <= 4:
            continue
        hits = 0.0
        for kt in kw_tokens:
            if kt in line_tokens:
                hits += 1.0
            elif len(kt) >= 5 and any(
                abs(len(lt) - len(kt)) <= 2 and SequenceMatcher(None, kt, lt).ratio() >= 0.78
                for lt in line_tokens
            ):
                hits += 0.85
        ratio = hits / len(kw_tokens)
        if ratio >= 0.75:
            best = max(best, ratio * 0.85)
    return best


def _line_text(line: List[OcrWord]) -> str:
    return " ".join(w.text for w in line).strip()


def _line_bbox(line: List[OcrWord]) -> BBox:
    x = min(w.bbox.x for w in line)
    y = min(w.bbox.y for w in line)
    x2 = max(w.bbox.x + w.bbox.w for w in line)
    y2 = max(w.bbox.y + w.bbox.h for w in line)
    return BBox(x, y, x2 - x, y2 - y)


def _window_indices(key: str, lines: List[List[OcrWord]], idx: int) -> List[int]:
    """Matched line + up to ADJACENT_LINE_WINDOW lines directly below it."""
    if key not in MULTI_LINE_WINDOW_FIELDS:
        return [idx]
    cur_bb = _line_bbox(lines[idx])
    page_w = max(w.bbox.x + w.bbox.w for ln in lines for w in ln)
    tol = X_ALIGN_TOLERANCE_RATIO * page_w
    chosen, last_bb = [idx], cur_bb
    if key == "MFR_ADDRESS" and PIN_RE.search(_line_text(lines[idx])):
        return chosen
    for j in range(idx + 1, len(lines)):
        if len(chosen) - 1 >= ADJACENT_LINE_WINDOW:
            break
        nb = _line_bbox(lines[j])
        if nb.y <= last_bb.y:
            continue                                   # same row / other column
        if nb.y - (last_bb.y + last_bb.h) > MAX_VERTICAL_GAP_RATIO * max(last_bb.h, 1.0):
            break                                      # too far below
        aligned = abs(nb.x - cur_bb.x) < tol or abs((nb.x + nb.w / 2) - (cur_bb.x + cur_bb.w / 2)) < tol
        if not aligned:
            continue
        chosen.append(j)
        last_bb = nb
        if key == "MFR_ADDRESS" and PIN_RE.search(_line_text(lines[j])):
            break
    return chosen


def _union_bbox(boxes: List[BBox]) -> BBox:
    x = min(b.x for b in boxes)
    y = min(b.y for b in boxes)
    x2 = max(b.x + b.w for b in boxes)
    y2 = max(b.y + b.h for b in boxes)
    return BBox(x, y, x2 - x, y2 - y)


def _build_extraction(key: str, lines, idx: int, conf: float) -> Tuple[FieldExtraction, List[int]]:
    ids = _window_indices(key, lines, idx)
    sel = [lines[i] for i in ids]
    boxes = [_line_bbox(l) for l in sel]
    text = " ".join(_line_text(l) for l in sel)
    # font height comes from the individual lines, never the union box
    font_px = float(statistics.median(b.h for b in boxes))
    return FieldExtraction(key, text, conf, _union_bbox(boxes), font_px), ids


def _rule_pass(lines: List[List[OcrWord]]) -> Tuple[Dict[str, FieldExtraction], Set[int]]:
    results: Dict[str, FieldExtraction] = {}
    claimed: Set[int] = set()

    for key, sig in FIELD_SIGNATURES.items():
        best = None
        for i, line in enumerate(lines):
            text = _line_text(line)
            score = (_keyword_score(text, sig["keywords"]) * sig["kw_weight"]
                     + (1.0 if sig["pattern"].search(text) else 0.0) * sig["pat_weight"])
            if score > 0.3 and (best is None or score > best[1]):
                best = (i, score)
        if best:
            ext, ids = _build_extraction(key, lines, best[0], min(best[1], 0.98))
            results[key] = ext
            claimed.update(ids)
        else:
            results[key] = FieldExtraction(key, None, 0.0, None, None)

    # COMMON_NAME heuristic: tallest alphabetic line not claimed by another field.
    best_name = None
    for i, line in enumerate(lines):
        text = _line_text(line)
        if i in claimed or len(text) < 3:
            continue
        if sum(c.isdigit() for c in text) / max(len(text), 1) > 0.3:
            continue
        bb = _line_bbox(line)
        if best_name is None or bb.h > best_name[1].h:
            best_name = (i, bb)
    if best_name:
        i, bb = best_name
        results["COMMON_NAME"] = FieldExtraction("COMMON_NAME", _line_text(lines[i]), 0.65, bb, bb.h)
    else:
        results["COMMON_NAME"] = FieldExtraction("COMMON_NAME", None, 0.0, None, None)
    return results, claimed


def classify_fields_rule_based(words: List[OcrWord]) -> List[FieldExtraction]:
    """Backward-compatible public wrapper (list in FIELD_ORDER)."""
    res, _ = _rule_pass(group_words_into_lines(words))
    return [res[k] for k in FIELD_ORDER]


# ---------------------------------------------------------------------------
# Optional fine-tuned DistilBERT line classifier
# ---------------------------------------------------------------------------

LINE_CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "distilbert-lmpc-ner")
_line_clf = None
_line_clf_attempted = False


def get_line_classifier():
    global _line_clf, _line_clf_attempted
    if _line_clf_attempted:
        return _line_clf
    _line_clf_attempted = True
    if os.environ.get("ENABLE_LINE_CLASSIFIER", "false").lower() not in ("1", "true", "yes"):
        return None
    if not os.path.isdir(LINE_CLASSIFIER_PATH):
        return None
    try:
        from transformers import pipeline
        _line_clf = pipeline("text-classification", model=LINE_CLASSIFIER_PATH)
    except Exception:
        _line_clf = None
    return _line_clf


def _apply_line_classifier(lines, rule: Dict[str, FieldExtraction]) -> None:
    clf = get_line_classifier()
    if clf is None or not lines:
        return
    try:
        preds = clf([_line_text(l) for l in lines], truncation=True, batch_size=16)
    except Exception:
        return
    best: Dict[str, Tuple[int, float]] = {}
    for i, p in enumerate(preds):
        p = p[0] if isinstance(p, list) else p
        label, score = p.get("label"), float(p.get("score", 0.0))
        # ignore un-named heads ("LABEL_3") and the NONE class
        if label in FIELD_ORDER and score >= 0.85 and (label not in best or score > best[label][1]):
            best[label] = (i, score)
    for key, (i, score) in best.items():
        cur = rule[key]
        if cur.extracted_text is None or cur.confidence < 0.5:
            rule[key], _ = _build_extraction(key, lines, i, min(0.6 + 0.3 * score, 0.9))


# ---------------------------------------------------------------------------
# VLM access + grounding
# ---------------------------------------------------------------------------

_vlm_state = {"checked_at": 0.0, "ok": False}
VLM_AVAILABILITY_TTL = 30.0


def vlm_enabled() -> bool:
    if os.environ.get("ENABLE_VLM", "true").lower() not in ("1", "true", "yes"):
        return False
    now = time.time()
    if now - _vlm_state["checked_at"] > VLM_AVAILABILITY_TTL:
        _vlm_state["ok"] = vlm_extractor.is_vlm_available()
        _vlm_state["checked_at"] = now
    return _vlm_state["ok"]


def fetch_vlm_fields(image_bgr: np.ndarray) -> Optional[Dict[str, Optional[str]]]:
    """Never raises. None => no VLM signal this run (rules take over)."""
    try:
        if not vlm_enabled():
            return None
        h, w = image_bgr.shape[:2]
        scale = VLM_MAX_SIDE / max(h, w)
        if scale < 1.0:
            image_bgr = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        result = vlm_extractor.extract_fields_vlm(image_bgr)
        if result is None:
            _vlm_state["checked_at"] = 0.0     # re-probe on the next scan
        return result
    except Exception:
        return None


def _tok_in(tok: str, pool: Set[str]) -> bool:
    if tok in pool:
        return True
    if len(tok) < 5 or any(c.isdigit() for c in tok):
        return False                            # numbers must match exactly
    return any(len(o) >= 4 and SequenceMatcher(None, tok, o).ratio() >= 0.85 for o in pool)


def _match_text_to_lines(text: str, lines) -> Optional[List[int]]:
    """Which OCR lines does this VLM string come from? Returns the indices of
    the best *contiguous* run of lines, or None when OCR doesn't corroborate it.

    A line qualifies when most of ITS tokens are in the VLM text (precision --
    e.g. an address line) or when the line contains most of the VLM value
    (recall -- e.g. VLM says "200 ml", line reads "Net Qty: 200 ml"). Taking
    only the best contiguous run stops a brand name that also appears in the
    address from stretching the box up to the top of the label."""
    t_tokens = set(_tokens(text))
    if not t_tokens:
        return None
    cand: Dict[int, Set[str]] = {}
    for i, line in enumerate(lines):
        lt = _tokens(_line_text(line))
        if not lt or (len(lt) == 1 and len(lt[0]) < 4 and not lt[0].isdigit()):
            continue
        hit = {t for t in lt if _tok_in(t, t_tokens)}
        vhit = {v for v in t_tokens if _tok_in(v, set(lt))}
        if hit and (len(hit) / len(lt) >= 0.6 or len(vhit) / len(t_tokens) >= 0.6):
            cand[i] = vhit
    if not cand:
        return None

    runs: List[List[int]] = []
    for i in sorted(cand):
        if runs and i - runs[-1][-1] <= 2:      # tolerate one unmatched line (OCR split)
            runs[-1].append(i)
        else:
            runs.append([i])

    def run_score(run):
        covered = set().union(*(cand[i] for i in run))
        return (len(covered), -len(run))         # more coverage, then tighter run

    best = max(runs, key=run_score)
    covered = set().union(*(cand[i] for i in best))
    return best if len(covered) / len(t_tokens) >= 0.5 else None


def _ground_vlm_field(key: str, vtext: str, lines) -> FieldExtraction:
    ids = _match_text_to_lines(vtext, lines)
    if not ids:
        return FieldExtraction(key, vtext, 0.85, None, None)     # font size unverifiable
    boxes = [_line_bbox(lines[i]) for i in ids]
    text = vtext
    if key == "MRP" and not TAX_CLAUSE_RE.search(text):
        # The tax clause often sits on the next printed line and the VLM
        # drops it; look at the neighbouring OCR lines before failing Rule 18.
        near = " ".join(_line_text(lines[i]) for i in range(max(0, min(ids) - 1), min(len(lines), max(ids) + 3)))
        m = TAX_CLAUSE_RE.search(near)
        if m:
            text = f"{text} {m.group(0)}"
    return FieldExtraction(key, text, 0.9, _union_bbox(boxes), float(statistics.median(b.h for b in boxes)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def classify_fields(
    words: List[OcrWord],
    raw_text: str,
    vlm_fields: Optional[Dict[str, Optional[str]]] = None,
) -> List[FieldExtraction]:
    """VLM-primary merge. `vlm_fields=None` => rules(+optional DistilBERT) only."""
    lines = group_words_into_lines(words)
    rule, _ = _rule_pass(lines)
    _apply_line_classifier(lines, rule)

    if vlm_fields is None:
        return [rule[k] for k in FIELD_ORDER]

    out: List[FieldExtraction] = []
    for key in FIELD_ORDER:
        vtext = vlm_fields.get(key)
        r = rule[key]
        if vtext:
            out.append(_ground_vlm_field(key, vtext, lines))
        elif r.extracted_text and (r.confidence >= RULE_FALLBACK_MIN_CONF or key == "COMMON_NAME"):
            r.confidence = min(r.confidence, 0.7)
            out.append(r)
        else:
            out.append(FieldExtraction(key, None, 0.0, None, None))
    return out
