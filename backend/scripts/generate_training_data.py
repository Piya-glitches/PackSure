"""
SYNTHETIC TRAINING DATA GENERATOR -- Phase 2 bootstrap for CRNN + DistilBERT.

Extends the same idea already used in app/pipeline/sample_labels.py (render
a label with PIL, so you know the ground truth because you drew it) but
parametrized to spit out hundreds/thousands of randomized variants instead
of 3 fixed demo images. Every variant produces TWO training signals at once,
for free, with zero manual labeling:

  1. CRNN data   -> per-line CROPPED IMAGE + the exact string drawn on it.
                    Saved to data/synthetic/lines/*.png + crnn_manifest.csv

  2. NER data    -> the same per-line STRING + which of your 8 LMPC field
                    classes (or NONE) it belongs to.
                    Saved to data/synthetic/ner_manifest.jsonl

Realism is added via noise augmentation (blur, jpeg artifacts, rotation,
brightness/contrast jitter, synthetic glare) since real phone photos are
never as clean as a freshly-rendered PIL image -- training only on clean
renders would not generalize to real photos.

USAGE:
    cd backend
    pip install --break-system-packages faker  # optional, falls back to
                                                 # built-in name lists if absent
    python scripts/generate_training_data.py --n 500

OUTPUT:
    data/synthetic/lines/*.png       -- individual text-line crops (CRNN input)
    data/synthetic/crnn_manifest.csv  -- filepath,text
    data/synthetic/ner_manifest.jsonl -- {"text": ..., "label": ...} per line
    data/synthetic/full/*.png         -- full label images (for eyeballing / PDP data later)

This script has NO dependency on the rest of the app package (no torch,
no transformers) so it runs fine on a plain laptop with no GPU. Training
itself (separate script, run on Colab) is what needs a GPU.
"""

import argparse
import csv
import json
import os
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import barcode
    from barcode.writer import ImageWriter
    HAVE_BARCODE = True
except Exception:
    HAVE_BARCODE = False

# ---------------------------------------------------------------------------
# Field vocabulary -- MUST stay in sync with app/pipeline/field_classifier.py
# FIELD_SIGNATURES keys, so the NER labels this script produces match the
# schema the real classifier (and eventually the fine-tuned model) uses.
# ---------------------------------------------------------------------------
FIELD_LABELS = [
    "MRP", "NET_QTY", "MFG_DATE", "MFR_ADDRESS",
    "CONSUMER_CARE", "COUNTRY_OF_ORIGIN", "UNIT_PRICE", "COMMON_NAME", "NONE",
]

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_FILES = [
    "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
    "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf",
    "DejaVuSansCondensed.ttf", "DejaVuSansMono.ttf",
]

# ---------------------------------------------------------------------------
# Random content pools -- small built-in lists so this works with zero
# external dependencies. Swap in `faker` for more variety if installed.
# ---------------------------------------------------------------------------
BRAND_WORDS = ["HERBAL GLOW", "CRUNCHIE BITES", "BUTTERKIST", "SUNRISE FOODS",
               "GOLDEN HARVEST", "NATURA FRESH", "TASTY BEST", "PUREWELL",
               "MOUNTAIN DAIRY", "SPICE ROUTE", "DAILY NEEDS", "FARM FRESH CO"]
PRODUCT_TYPES = ["Shampoo 200ml", "Namkeen Mixture", "Butter Cookies 150g",
                  "Toor Dal 1kg", "Instant Noodles", "Face Wash 100ml",
                  "Peanut Butter 340g", "Basmati Rice 5kg", "Green Tea Bags",
                  "Multigrain Atta 5kg", "Coconut Oil 500ml", "Sugar 1kg"]
CITIES = ["Pune - 411018", "Indore - 452001", "Gurugram - 122001",
          "Anand - 388001", "Nashik - 422001", "Rajkot - 360001",
          "Ludhiana - 141001", "Coimbatore - 641001", "Bhopal - 462001",
          "Vadodara - 390001"]
STREETS = ["Plot 14, MIDC", "Sector 12", "At Food Complex Mogar", "Industrial Area Phase 2",
           "GIDC Estate", "Survey No. 45", "Shed No. 7, Co-op Estate"]
COMPANY_SUFFIX = ["Pvt Ltd", "Foods", "Industries Ltd", "Co-operative Union Ltd",
                   "Enterprises", "Agro Products Pvt Ltd"]
MRP_QUALIFIER = ["(Incl. of all taxes)", "(inclusive of all taxes)", ""]  # "" -> intentionally non-compliant
UNITS = ["g", "kg", "ml", "l", "pieces"]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


def rand_phone():
    return f"1800-{random.randint(100,999)}-{random.randint(1000,9999)}"


def rand_email(brand: str):
    slug = "".join(c for c in brand.lower() if c.isalnum())[:10]
    return f"care@{slug}.in"


def rand_pin():
    return random.choice(CITIES)


def make_field_lines():
    """Returns list of (text, field_label, font_size_mm, bold) with randomized
    content. Occasionally DROPS or MALFORMS a field so negative/edge examples
    exist too (matches your real compliance-failure modes)."""
    brand = random.choice(BRAND_WORDS)
    product = random.choice(PRODUCT_TYPES)
    company = f"{brand.title()} {random.choice(COMPANY_SUFFIX)}"
    net_val = random.randint(50, 999)
    unit = random.choice(UNITS)
    mrp_val = random.randint(20, 999)
    qualifier = random.choice(MRP_QUALIFIER)
    month, year = random.choice(MONTHS), random.randint(24, 27)

    lines = [
        (brand, "NONE", 6, True),
        (product, "COMMON_NAME", 5, True),
        (f"Net Qty: {net_val} {unit}", "NET_QTY", 3, False),
        (f"MRP: Rs. {mrp_val}.00 {qualifier}".strip(), "MRP", 3, False),
        (f"Mfg Date: {month}/{year+2000}", "MFG_DATE", 3, False),
        (f"Manufactured by: {company},", "MFR_ADDRESS", 2.5, False),
        (f"{random.choice(STREETS)}, {rand_pin()}", "MFR_ADDRESS", 2.5, False),
        (f"Customer Care: {rand_phone()}, {rand_email(brand)}", "CONSUMER_CARE", 2.5, False),
        ("Country of Origin: India", "COUNTRY_OF_ORIGIN", 2.5, False),
        (f"Unit Sale Price: Rs. {mrp_val/max(net_val,1)*100:.2f} / 100{unit}", "UNIT_PRICE", 2.5, False),
    ]

    # Randomly drop 0-2 optional fields to create realistic non-compliant/
    # missing-field examples (mirrors real-world label variance).
    if random.random() < 0.3:
        drop_idx = random.randint(4, len(lines) - 1)
        lines.pop(drop_idx)

    return lines


def _load_font(size_px: int, bold: bool = False):
    candidates = [f for f in FONT_FILES if ("Bold" in f) == bold] or FONT_FILES
    path = os.path.join(FONT_DIR, random.choice(candidates))
    try:
        return ImageFont.truetype(path, max(size_px, 8))
    except Exception:
        return ImageFont.load_default()


def _mm(v, px_per_mm=6):
    return int(v * px_per_mm)


def render_label(sample_idx: int):
    """Renders one full synthetic label. Returns (full_image_bgr, line_records)
    where line_records is a list of dicts with text/label/bbox in pixel coords
    on the CLEAN (pre-noise) image -- we crop before adding geometric noise so
    crops stay valid."""
    px_per_mm = 6
    width_mm, height_mm = 80, 115
    img = Image.new("RGB", (_mm(width_mm, px_per_mm), _mm(height_mm, px_per_mm)), "#FDFBF6")
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, img.width - 3, img.height - 3], outline="#CBB68A", width=2)

    lines = make_field_lines()
    cursor_y = _mm(8, px_per_mm)
    records = []
    for text, field_label, font_size_mm, bold in lines:
        font = _load_font(_mm(font_size_mm, px_per_mm), bold)
        x = _mm(6, px_per_mm)
        bbox = draw.textbbox((x, cursor_y), text, font=font)
        draw.text((x, cursor_y), text, fill="#1A1A1A", font=font)
        records.append({"text": text, "label": field_label, "bbox": bbox})
        cursor_y += int(_mm(font_size_mm, px_per_mm) * 1.7)

    if HAVE_BARCODE:
        try:
            digits = "".join(random.choice(string.digits) for _ in range(12))
            ean = barcode.get("ean13", digits, writer=ImageWriter())
            import io
            buf = io.BytesIO()
            ean.write(buf, options={"module_width": 0.33, "module_height": 15,
                                     "quiet_zone": 2.0, "font_size": 8, "write_text": True})
            buf.seek(0)
            bc_img = Image.open(buf).convert("RGB")
            bw, bh = bc_img.size
            scale = _mm(37.29, px_per_mm) / bw
            bc_img = bc_img.resize((int(bw * scale), int(bh * scale)))
            img.paste(bc_img, (_mm(6, px_per_mm), cursor_y + _mm(4, px_per_mm)))
        except Exception:
            pass

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), records


def add_noise(image_bgr: np.ndarray) -> np.ndarray:
    """Applies randomized realistic degradation so the model doesn't only
    ever see pixel-perfect renders. Order matters (blur before jpeg, etc.)."""
    img = image_bgr.copy()

    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    alpha = random.uniform(0.8, 1.25)   # contrast
    beta = random.uniform(-20, 20)      # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.35:
        h, w = img.shape[:2]
        overlay = img.copy()
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.randint(int(0.15 * w), int(0.4 * w))
        cv2.circle(overlay, (cx, cy), r, (255, 255, 255), -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        quality = random.randint(35, 80)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of synthetic labels to generate")
    ap.add_argument("--out", type=str, default="data/synthetic", help="output directory")
    ap.add_argument("--save-full", action="store_true", help="also save full label images (bigger, useful later for PDP/YOLO seed data)")
    args = ap.parse_args()

    lines_dir = os.path.join(args.out, "lines")
    full_dir = os.path.join(args.out, "full")
    os.makedirs(lines_dir, exist_ok=True)
    if args.save_full:
        os.makedirs(full_dir, exist_ok=True)

    crnn_rows = []
    ner_rows = []

    for i in range(args.n):
        clean_img, records = render_label(i)

        if args.save_full:
            noisy_full = add_noise(clean_img)
            cv2.imwrite(os.path.join(full_dir, f"label_{i:05d}.png"), noisy_full)

        for j, rec in enumerate(records):
            x0, y0, x1, y1 = rec["bbox"]
            pad = 3
            h, w = clean_img.shape[:2]
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
            crop = clean_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = add_noise(crop)

            fname = f"line_{i:05d}_{j:02d}.png"
            cv2.imwrite(os.path.join(lines_dir, fname), crop)

            crnn_rows.append({"filepath": os.path.join("lines", fname), "text": rec["text"]})
            ner_rows.append({"text": rec["text"], "label": rec["label"]})

    with open(os.path.join(args.out, "crnn_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "text"])
        writer.writeheader()
        writer.writerows(crnn_rows)

    with open(os.path.join(args.out, "ner_manifest.jsonl"), "w", encoding="utf-8") as f:
        for row in ner_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = {}
    for row in ner_rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    print(f"Generated {args.n} synthetic labels -> {len(crnn_rows)} text-line examples")
    print(f"CRNN manifest: {os.path.join(args.out, 'crnn_manifest.csv')}")
    print(f"NER manifest:  {os.path.join(args.out, 'ner_manifest.jsonl')}")
    print("Label distribution:", label_counts)
    if not HAVE_BARCODE:
        print("NOTE: python-barcode not installed -- labels generated without barcodes. "
              "pip install --break-system-packages python-barcode to include them.")


if __name__ == "__main__":
    main()
