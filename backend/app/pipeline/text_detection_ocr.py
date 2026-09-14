"""
LAYER 2 -- MULTI-SCRIPT TEXT DETECTION & OCR

Per the architecture, Layer 2 needs: CRAFT (Character Region Awareness for
Text) for detection, and a CRNN (CNN + BiLSTM + CTC) for recognition,
fine-tuned/capable on Indian packaging with Hindi+English support.

We use EasyOCR to satisfy this layer -- and this is a genuine architectural
match, not a substitution: EasyOCR's detector IS CRAFT, and EasyOCR's
default recognizer IS a CRNN (a VGG/ResNet feature extractor -> BiLSTM ->
CTC decoder -- exactly the CNN+BiLSTM+CTC structure named in the
architecture doc), pretrained on real data and shipping with an 'hi'
(Hindi) + 'en' (English) language pack out of the box. Where Tesseract is
a classical LSTM-per-character engine with no CRAFT-style detector at all,
EasyOCR is a direct, pretrained implementation of the exact two-stage
CRAFT+CRNN pipeline this layer specifies.

See crnn_model.py for the standalone CRNN architecture definition used for
the Phase 2 fine-tuning roadmap item (training on an OCR-error-augmented,
Indian-packaging-specific dataset, as the architecture's "noise tolerance"
requirement calls for).
"""

from typing import List
import numpy as np
import easyocr

from app.pipeline.types import OcrWord, BBox

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        # 'en' + 'hi' matches the architecture's explicit Hindi+English
        # requirement. gpu=False for portability; set True if a CUDA GPU
        # is available on the deploy target for materially faster inference.
        _reader = easyocr.Reader(["en", "hi"], gpu=False)
    return _reader


def run_ocr(image_bgr: np.ndarray) -> tuple[str, List[OcrWord]]:
    reader = get_reader()
    # detail=1 returns (bbox_points, text, confidence) per detected region --
    # this is the CRAFT detector's region proposals each passed through the
    # CRNN recognizer, exactly the two-stage pipeline the architecture calls for.
    results = reader.readtext(image_bgr, detail=1, paragraph=False)

    words: List[OcrWord] = []
    text_parts: List[str] = []

    for bbox_points, text, confidence in results:
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y
        words.append(OcrWord(text=text, confidence=float(confidence), bbox=BBox(x, y, w, h)))
        text_parts.append(text)

    raw_text = "\n".join(text_parts)
    return raw_text, words


def group_words_into_lines(words: List[OcrWord]) -> List[List[OcrWord]]:
    """Groups OCR word-regions into lines by vertical proximity, since the
    field classifier (Layer 3) reasons over phrases, not isolated words --
    legal declarations like addresses and MRP statements span several
    words/regions."""
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
