"""
Is the VLM (Qwen2.5-VL via Ollama) reachable, is the model pulled, and does it return the 8-field JSON?

Run from backend/:   python scripts/check_vlm.py
Uses OLLAMA_URL / VLM_MODEL_NAME from backend/.env.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config  # noqa: F401,E402  (loads .env first)

import numpy as np  # noqa: E402
import requests  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.pipeline import vlm_extractor as vx  # noqa: E402

url, model = vx.OLLAMA_URL, vx.VLM_MODEL_NAME
print(f"Ollama URL : {url}\nModel      : {model}\n")

try:
    tags = requests.get(f"{url}/api/tags", timeout=8)
    tags.raise_for_status()
except requests.RequestException as e:
    sys.exit(f"[FAIL] cannot reach Ollama: {e}\n"
             "  - local: is `ollama serve` running?\n"
             "  - Colab tunnel: is the trycloudflare URL current (it changes every session)?\n"
             "  - Docker: use http://host.docker.internal:11434 and rebuild compose (extra_hosts)")

names = [m["name"] for m in tags.json().get("models", [])]
print("models on server:", names or "(none)")
if model not in names and f"{model}:latest" not in names:
    sys.exit(f"[FAIL] model '{model}' is not pulled.  Run:  ollama pull {model}")
print("[ok] server reachable, model present\n")

# Render a small synthetic label and ask the model to read it.
try:
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
except Exception:
    f = fb = ImageFont.load_default()
img = Image.new("RGB", (1000, 720), "white")
d = ImageDraw.Draw(img)
d.text((40, 30), "HERBAL GLOW", fill="black", font=fb)
d.text((40, 110), "Shampoo", fill="black", font=fb)
for i, t in enumerate(["Net Qty: 200 ml", "MRP: Rs. 199.00 (Incl. of all taxes)", "Mfg Date: 03/2026",
                       "Manufactured by: Herbal Glow Pvt Ltd,", "Plot 14, MIDC, Pune - 411018",
                       "Customer Care: 1800-102-3456", "Country of Origin: India"]):
    d.text((40, 220 + i * 60), t, fill="black", font=f)
bgr = np.array(img)[:, :, ::-1].copy()

t0 = time.time()
out = vx.extract_fields_vlm(bgr)
dt = time.time() - t0
if out is None:
    sys.exit(f"[FAIL] no usable JSON after {dt:.0f}s (timeout={vx.VLM_TIMEOUT_SECONDS}s?). "
             "On CPU, first call loads the model; retry once. Otherwise use a GPU host (see MIGRATION.md).")
print(f"[ok] VLM answered in {dt:.1f}s:")
for k, v in out.items():
    print(f"   {k:18s} {v!r}")
expected = {"NET_QTY": "200", "MFG_DATE": "03/2026", "COUNTRY_OF_ORIGIN": "India"}
bad = [k for k, s in expected.items() if not (out.get(k) and s.lower() in out[k].lower())]
print("\nSANITY:", "all expected fields read correctly" if not bad else f"misread: {bad} (try a larger model, e.g. qwen2.5vl:7b)")
