"""
Built-in sample labels let a judge/demo exercise the FULL pipeline
(barcode calibration -> PDP detection -> OCR -> field classification ->
font-size-in-mm -> rule validation -> responsible-party resolution) with
zero uploads. Uses python-barcode to render a REAL, checksum-valid,
scannable EAN-13 barcode (decodable by pyzbar, the same library the live
pipeline uses) -- not a fake striped rectangle.

We draw at a KNOWN, self-chosen scale (6 px/mm) purely to construct the
synthetic image; the pipeline must still independently discover that
scale via the barcode, exactly as it would for a real photo. That's what
makes this a genuine end-to-end test rather than a rigged shortcut.
"""

import io
from typing import Literal

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import barcode
from barcode.writer import ImageWriter

PX_PER_MM = 6


def _mm(v: float) -> int:
    return int(v * PX_PER_MM)


def _make_ean13_image(digits12: str, module_width_mm: float = 0.33, height_mm: float = 15) -> Image.Image:
    ean = barcode.get("ean13", digits12, writer=ImageWriter())
    buf = io.BytesIO()
    ean.write(
        buf,
        options={
            "module_width": module_width_mm,
            "module_height": height_mm,
            "quiet_zone": 2.0,
            "font_size": 8,
            "text_distance": 2,
            "write_text": True,
        },
    )
    buf.seek(0)
    return Image.open(buf).convert("RGB")


SAMPLE_SPECS = {
    "compliant-shampoo": {
        "title": "Compliant — Herbal Shampoo 200ml",
        "expected": "COMPLIANT",
        "barcode_digits": "890123456789",
        "lines": [
            ("HERBAL GLOW", 6, True),
            ("Shampoo", 5, True),
            ("Net Qty: 200 ml", 3, False),
            ("MRP: Rs. 199.00 (Incl. of all taxes)", 3, False),
            ("Mfg Date: 03/2026", 3, False),
            ("Manufactured by: Herbal Glow Pvt Ltd,", 2.5, False),
            ("Plot 14, MIDC, Pune - 411018", 2.5, False),
            ("Customer Care: 1800-102-3456, care@herbalglow.in", 2.5, False),
            ("Country of Origin: India", 2.5, False),
            ("Unit Sale Price: Rs. 99.50 / 100ml", 2.5, False),
        ],
    },
    "missing-fields-snack": {
        "title": "Non-Compliant — Namkeen Snack Pack",
        "expected": "NON_COMPLIANT",
        "barcode_digits": "890198765432",
        "lines": [
            ("CRUNCHIE BITES", 6, True),
            ("Namkeen Mixture", 5, True),
            ("Net Qty: 150 g", 3, False),
            ("MRP: Rs. 40.00", 3, False),  # missing "inclusive of all taxes"
            ("Pkd Date: 07/2026", 3, False),
            ("Packed by: Crunchie Foods, Indore - 452001", 2.5, False),
            ("Country of Origin: India", 2.5, False),
            # consumer care intentionally omitted
        ],
    },
    "small-font-cookies": {
        "title": "Needs Review — Butter Cookies 150g",
        "expected": "NEEDS_REVIEW",
        "barcode_digits": "890111222333",
        "lines": [
            ("BUTTERKIST", 6, True),
            ("Butter Cookies", 5, True),
            ("Net Qty: 150 g", 0.7, False),  # deliberately undersized
            ("MRP: Rs. 60.00 (Incl. of all taxes)", 3, False),
            ("Mfg Date: 01/2026", 3, False),
            ("Manufactured by: Butterkist Bakers,", 2.5, False),
            ("Sector 12, Gurugram - 122001", 2.5, False),
            ("Customer Care: 1800-555-2211", 2.5, False),
            ("Country of Origin: India", 2.5, False),
        ],
    },
}


def _load_font(size_px: int, bold: bool = False):
    try:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        return ImageFont.truetype(path, size_px)
    except Exception:
        return ImageFont.load_default()


def generate_sample_label(sample_id: str) -> np.ndarray:
    """Returns a BGR numpy array (OpenCV format), ready to feed straight
    into run_full_pipeline exactly like a real uploaded photo."""
    spec = SAMPLE_SPECS[sample_id]

    width_mm, height_mm = 80, 110
    img = Image.new("RGB", (_mm(width_mm), _mm(height_mm)), "#FDFBF6")
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, img.width - 3, img.height - 3], outline="#CBB68A", width=2)

    cursor_y = _mm(8)
    for text, font_size_mm, bold in spec["lines"]:
        font = _load_font(_mm(font_size_mm), bold)
        draw.text((_mm(6), cursor_y), text, fill="#1A1A1A", font=font)
        cursor_y += int(_mm(font_size_mm) * 1.6)

    barcode_img = _make_ean13_image(spec["barcode_digits"])
    bw, bh = barcode_img.size
    scale = _mm(37.29) / bw  # scale rendered barcode to our chosen physical width
    barcode_img = barcode_img.resize((int(bw * scale), int(bh * scale)))
    img.paste(barcode_img, (_mm(6), cursor_y + _mm(4)))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def list_samples():
    return [{"id": k, "title": v["title"], "expected_outcome": v["expected"]} for k, v in SAMPLE_SPECS.items()]
