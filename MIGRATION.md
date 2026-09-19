# PackSure — migration to the `systechchange.pdf` stack

## 1. File map (repo paths → what to do)

| Repo path | Action | Source |
|---|---|---|
| `backend/app/pipeline/barcode_calibration.py` | replace | your refined file (unchanged) |
| `backend/app/pipeline/dewarp.py` | replace | your refined file (unchanged) |
| `backend/app/pipeline/text_detection_ocr.py` | replace | your refined file (unchanged) |
| `backend/app/pipeline/vlm_extractor.py` | add | your refined file (unchanged) |
| `backend/app/pipeline/pdp_segmentation.py` | **replace** | **this bundle** — your file + a 2-line `order_points` fix (see §4) |
| `backend/app/pipeline/compliance_engine.py` | **replace** | this bundle (rewritten) |
| `backend/app/pipeline/field_classifier.py` | **replace** | this bundle (rewritten) |
| `backend/app/pipeline/quality_gate.py` | **replace** | this bundle |
| `backend/app/pipeline/sample_labels.py` | **replace** | this bundle |
| `backend/app/config.py`, `backend/app/main.py` | **replace** | this bundle |
| `backend/requirements.txt`, `backend/Dockerfile`, `backend/.env.example`, `docker-compose.yml` | **replace** | this bundle |
| `backend/scripts/run_on_image.py`, `backend/scripts/evaluate.py` | add | this bundle |
| `training/train_yolov8_seg_colab.ipynb` | add | this bundle |
| `backend/app/pipeline/pdp_detection.py` | **delete** | replaced by `pdp_segmentation.py` |
| `backend/app/pipeline/crnn_model.py` | **delete** | the doc says don't train a CRNN from scratch |
| `backend/.gitignore` | append `models/*.onnx` | |
| `types.py`, `font_size.py`, `rule_validators.py`, `responsible_party.py`, `routers/*`, `models.py`, `pdf_report.py`, whole `frontend/` | **no change** | |

Do **not** use the repo-root `requirements.txt` from your context — it has unpinned `paddleocr>=2.7.3`/`paddlepaddle>=2.6.0`,
which now resolve to PaddleOCR **3.x**. 3.x removed `use_gpu`, `use_angle_cls`, `show_log` and `.ocr(cls=True)` and changed the
result format, so your `text_detection_ocr.py` would fail at startup. `backend/requirements.txt` pins the 2.x line
(`paddlepaddle==2.6.2`, `paddleocr==2.10.0`). If pip says a pin doesn't exist for your Python, tell me the error and I'll adjust.

## 2. Run it

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload         # /health shows which models loaded
python scripts/run_on_image.py path/to/real_pack.jpg          # run from backend/, not scripts/
```
It works with **zero trained models**: contour fallback for the label, PaddleOCR for text, rule-based extraction if no VLM.
Every trained/served component only upgrades a stage; each stage reports which path ran (`processing_log`, `pdp_detection_method`).

## 3. Serving Qwen2.5-VL (Layer 3, the biggest accuracy jump) — on a free Colab T4

A 3B VLM on a laptop CPU takes minutes per label. On a T4 it takes seconds. Run Ollama in Colab and tunnel it:

```python
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time, os
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
subprocess.Popen(["ollama", "serve"]); time.sleep(5)
!ollama pull qwen2.5vl:3b
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
subprocess.Popen(["./cloudflared","tunnel","--url","http://localhost:11434","--http-host-header","localhost:11434"])
# copy the https://xxxx.trycloudflare.com URL printed in the logs
```
Then in `backend/.env`: `OLLAMA_URL=https://xxxx.trycloudflare.com`. The URL changes every Colab session. (I couldn't run this
in my sandbox — no network — so treat the snippet as a starting point.) If Ollama returns 403 through the tunnel, the
`--http-host-header` flag is the fix.

If the VLM is down, scans silently fall back to rules and the `vlm_wait` log line says so.

## 4. Bugs I found while wiring this — please read

1. **`order_points()` swapped top-right and bottom-left** (in your `pdp_segmentation.py` *and* the old `pdp_detection.py`).
   `np.diff` gives `y − x`, so top-right is the *min*, not the max. The homography was therefore *transposing* every
   contour-detected label before OCR. Fixed. This alone can explain a lot of "OCR reads garbage" results.
2. **Barcode scale was measured in the wrong pixel space.** The barcode is decoded on the raw photo (correct, per the doc), but
   OCR boxes live in the dewarped 1200-px crop, so raw px/mm made every font size wrong by the crop's zoom factor. The engine now
   carries the scale through the same transform (local Jacobian for homography, arcsin-aware for the cylinder path) and maps the
   barcode box too. I verified both against your real `dewarp.py` output on synthetic images: position within ~1 px, width within 2–4 %.
   If the raw decode fails, it retries on the dewarped pre-CLAHE image (your earlier "Bug #2" fix), which needs no conversion.
3. **Quality gate flagged white labels as "overexposed"** (`mean > 235`). Now also requires low contrast. Blur threshold 60 → 40,
   measured on the dewarped label *before* CLAHE. Both tunable; `evaluate.py` prints the numbers.
4. **Sample-label barcode scale** matched the old 37.29 mm constant (quiet zones included); now exactly 31.35 mm bar pattern to match `barcode_calibration.py`.
5. `.env` values weren't visible to `vlm_extractor.py` (pydantic-settings doesn't export to `os.environ`); `config.py` now calls `load_dotenv()`.
6. PaddleOCR predictors aren't thread-safe and FastAPI runs sync routes in a threadpool → OCR is behind a lock.
7. The old "DistilBERT backbone" loaded the *generic* pretrained checkpoint, whose classification head is random — it added no signal. Removed.
   Your fine-tuned line classifier plugs in as an optional third signal (`ENABLE_LINE_CLASSIFIER=true`, checkpoint in `models/distilbert-lmpc-ner/`; label names must be the field names).

Behaviour changes to be aware of: with no barcode found, a scan that would have been COMPLIANT is now NEEDS_REVIEW (font size was
unverifiable, so it shouldn't be stamped compliant). OCR and the VLM run in parallel.

## 5. What to train, in order

1. **Nothing, first.** Collect 30–50 real photos + a `labels.json` and run `python scripts/evaluate.py --images … --labels … --mode both`.
   It prints per-field accuracy for rules vs. VLM plus how often the barcode is found and how often YOLO-seg is used. Train what the numbers point at.
2. **YOLOv8n-seg (`training/train_yolov8_seg_colab.ipynb`)** — the only model the doc's roadmap really needs data for. One class `label_pdp`
   (the doc lists both one class and two classes; the delivered `pdp_segmentation.py` takes the top mask regardless of class, so two classes would let a barcode win).
   400+ real photos, polygon annotation, ~15 min on a T4. Drop `pdp_yolov8n_seg.onnx`/`.pt` in `backend/models/`.
3. **LayoutLMv3 / PaddleOCR-SVTR fine-tuning: skip for now.** The VLM removes the need for 1,000 word-level-annotated labels, and PP-OCR
   is already trained for Devanagari + Latin; fine-tune the recognizer only if `evaluate.py` shows OCR character errors on real crops.
4. Barcode-class detector (doc's "YOLO crops damaged barcodes"): only if `evaluate.py` shows a low calibration rate on your photos.

## 6. Known limits (say these before a judge does)

- **Barcode = ruler assumes 100 % GS1 magnification.** Real EAN-13s print at 80–200 %, so mm values are approximate; the code says so in `barcode_calibration.py`.
  The barcode must also lie on the same plane as the text you measure.
- **Please verify `MIN_FONT_SIZE_MM_BY_AREA` in `types.py`.** It tiers by PDP area (1/2/4 mm). My recollection is that Rule 9 keys
  numeral height to *net quantity* (roughly 1 / 2 / 4 / 6 mm bands) — I could not check the gazette here, so confirm before presenting it as statute.
- VLM values are grounded on OCR lines to get a box; if OCR doesn't corroborate the text, the field keeps the VLM text but its font size is reported unverifiable.
- `README.md` still describes EasyOCR/pyzbar/DistilBERT — update it before judging.
- I could not install zxing-cpp/PaddleOCR/ultralytics in my sandbox (no network). I tested the geometry, field logic and orchestration
  with stubs for those three; `run_on_image.py` on a real photo is the true integration test.
