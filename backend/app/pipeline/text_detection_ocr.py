"""
LAYER 2 -- MULTI-SCRIPT TEXT DETECTION & OCR

MIGRATED per systechchange.pdf "Technology Replacement Matrix": EasyOCR
-> PaddleOCR (PP-OCRv4/v5). EasyOCR's CRAFT detector struggles with dense
small fonts (exactly what's printed on an LMPC declaration block), and its
generic CRNN recognizer was never fine-tuned for Indian packaging fonts.
PaddleOCR's DBNet++ detector + SVTR recognizer are the current
production-grade standard for this: native bilingual English/Devanagari
support, and runs sub-100ms on ONNX Runtime once exported.

`lang="hi"` (rather than "en") is intentional -- PaddleOCR's Hindi model
pack is trained to recognise both Devanagari AND Latin-script English
digits/text in the same pass (unlike EasyOCR, which needed the two
language codes combined). If you find English-heavy labels come out
worse under 'hi' than under 'en' in practice, that's a real empirical
question to test on your actual photos, not a settled fact -- keep
LANG configurable below rather than hardcoded in two places.

Interface is UNCHANGED from the EasyOCR version on purpose: run_ocr()
still returns (raw_text: str, words: List[OcrWord]), and
group_words_into_lines() is untouched -- so field_classifier.py,
font_size.py, and compliance_engine.py need ZERO changes for this swap.
"""

from typing import List
import numpy as np

from app.pipeline.types import OcrWord, BBox

LANG = "hi"  # bilingual en+hi recognition pack

_ocr_engine = None


def get_reader():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        # use_angle_cls handles upside-down/rotated label crops (common when
        # a PDP quad comes out slightly mis-oriented from Layer 1).
        # show_log kwarg was removed in newer paddleocr versions; wrapped in
        # try/except so this works across the 2.7.x -> 2.8.x/3.x API drift.
        try:
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang=LANG, use_gpu=False, show_log=False)
        except TypeError:
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang=LANG, use_gpu=False)
    return _ocr_engine


def run_ocr(image_bgr: np.ndarray) -> tuple[str, List[OcrWord]]:
    engine = get_reader()

    # PaddleOCR's .ocr() expects RGB; the rest of the pipeline works in BGR
    # (OpenCV convention) throughout, so convert only at this boundary.
    import cv2
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = engine.ocr(image_rgb, cls=True)

    words: List[OcrWord] = []
    text_parts: List[str] = []

    # result is [[ [box, (text, score)], ... ]] -- one inner list per image.
    # Guard against None (PaddleOCR returns [None] for a blank/unreadable
    # crop rather than raising) and against API variants that drop the
    # outer wrapping list.
    lines = (result[0] if result and result[0] is not None else []) if result else []

    for entry in lines:
        box, (text, score) = entry[0], entry[1]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        words.append(OcrWord(text=text, confidence=float(score), bbox=BBox(x, y, w, h)))
        text_parts.append(text)

    raw_text = "\n".join(text_parts)
    return raw_text, words


def group_words_into_lines(words: List[OcrWord]) -> List[List[OcrWord]]:
    """UNCHANGED from the EasyOCR version -- PaddleOCR returns word/phrase-
    level boxes just like EasyOCR did, so the same vertical-proximity
    grouping logic applies without modification. field_classifier.py's
    line-level reasoning (addresses/MRP spanning several boxes) still works
    exactly as before."""
    sorted_words = sorted(words, key=lambda w: w.bbox.y)
    lines: List[List[OcrWord]] = []
    tolerance_ratio = 0.6

    for word in sorted_words:
        placed = False
        for line in lines:
            ref = line[0]
            tolerance = max(ref.bbox.h, word.bbox.h) * tolerance_ratio
            if abs(ref.bbox.y - word.bbox.y) <= tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    for line in lines:
        line.sort(key=lambda w: w.bbox.x)
    return lines
