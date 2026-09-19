# PackSure — training data + model integration, step by step

Only **one** model needs training data: the YOLOv8n-seg label detector. Everything else is pretrained.

| Component | Trained by you? | Integration |
|---|---|---|
| zxing-cpp (barcode) | no model at all | `pip install` — done |
| PaddleOCR (det + rec + angle) | **no — pretrained**, auto-downloads | first start downloads to `~/.paddleocr` |
| Qwen2.5-VL-3B (field extraction) | **no — pretrained**, served by Ollama | `ollama pull` + `OLLAMA_URL` (Part 4) |
| YOLOv8n-seg (label mask) | **yes** — ~400+ real photos (Parts 1–3) | drop 2 files in `backend/models/` (Part 3) |

All commands assume the repo layout from `MIGRATION.md`; data tools live in `training/data_tools/`.

---

## Part 1 — Build the YOLO-seg dataset

### 1.1 Photograph (the part that decides quality)
- **400–600 photos**, at least **60–100 different products**, 4–6 shots each: straight-on, ±30° angle, glare, dim light, shelf clutter, hand-held.
- Include every shape you'll meet: cartons, pouches, jars, **bottles and cans (curved)**, and some with no clear label.
- Keep each product's photos in its own folder: `raw/amul_butter/`, `raw/parle_g/` …
- Keep **30–50 extra photos out of training** for the evaluation set (Part 2). Shoot these at full phone resolution.

### 1.2 Clean the photos (once per product folder)
```bash
cd training/data_tools
python 01_prepare_photos.py --in raw/amul_butter --out photos_clean --prefix amul_butter
python 01_prepare_photos.py --in raw/parle_g     --out photos_clean --prefix parle_g
#   ... one call per product. Fixes EXIF rotation, drops near-duplicates, flags blurry shots.
```
Look at the blurry ones it lists and retake them.

### 1.3 Annotate — draw ONE polygon per photo around the printed label
```bash
pip install labelme
labelme photos_clean/ --labels label_pdp --autosave
```
- *Create Polygon* → click around the **label / principal display panel** (the face carrying MRP, net quantity, address).
  4 clicks for a box; 8–12 for a pouch or curved bottle label. Name it exactly `label_pdp`.
- Outline the label's **outer edge**, not the whole pack, not the shelf. Ignore neighbouring products.
- Budget: ~20–30 s per photo → 400 photos ≈ 3 hours. (Roboflow *Smart Polygon* is faster; if you use it, export **YOLOv8** and skip 1.4.)

### 1.4 Convert to YOLO format
```bash
python 02_labelme_to_yolo_seg.py --in photos_clean --out real
#   add --keep-negatives to keep photos with no label as empty-label negatives
```

### 1.5 (Optional) Add synthetic data
```bash
python 03_make_synthetic.py --out synth --count 1500 --seed 1
python 03_make_synthetic.py --out synth --count 1500 --backgrounds raw_shelf_photos/   # better: real backgrounds
```
Auto-labelled flat + cylindrically-wrapped packs on cluttered backgrounds. I checked the outlines visually — they hug the label,
including curved ones — but the images look cartoonish. **It helps most when you have < 200 real photos**; it does not replace real photos.

### 1.6 Assemble, split, verify
```bash
python 04_build_dataset.py --real real --synthetic synth --out dataset --val 0.15 --test 0.15 --group-by-prefix
python 05_check_dataset.py --dataset dataset --sheet check_overlay.jpg
```
- `--group-by-prefix` keeps all photos of one product in one split (otherwise near-identical shots leak from train into valid and inflate mAP).
- Synthetic images go into **train only**. Valid/test are real, so the score means something.
- **Open `check_overlay.jpg`.** Green outlines must hug the label. Fix bad annotations now — it's the cheapest debugging you'll do.
- Result: `dataset/` and `dataset.zip`.

---

## Part 2 — Evaluation set (tells you what's actually broken)

```bash
cd backend
python ../training/data_tools/01_prepare_photos.py --in raw_eval --out data/eval --max-side 0 --prefix eval
python scripts/make_eval_labels.py --images data/eval --out data/eval/labels.draft.json   # slow first run: loads models
#   open labels.draft.json, CORRECT every value against the photo, null = not printed, save as labels.json
python scripts/evaluate.py --images data/eval --labels data/eval/labels.json --mode both
```
`--max-side 0` keeps full resolution (barcode calibration needs it). Skipping the correction step makes the score meaningless — the draft is the pipeline's own output.
The report gives per-field accuracy for rules vs. VLM, barcode-found rate, YOLO usage, and blur numbers. Re-run it after every change.

---

## Part 3 — Train YOLO in Colab and integrate it

1. Upload `dataset.zip` to Google Drive → `MyDrive/packsure/dataset.zip`.
2. Open `training/train_yolov8_seg_colab.ipynb` in Colab → **Runtime → T4 GPU** → run all cells (~15–25 min).
   The dataset cell fails loudly if labels are boxes instead of polygons or if there's more than one class.
3. Read the **mask mAP50** it prints for the *real* validation set: ≥ 0.85 is good; < 0.7 usually means more/more varied photos, not more epochs.
4. Files appear in `MyDrive/packsure/models/`: `pdp_yolov8n_seg.onnx` and `pdp_yolov8n_seg.pt`.

**Integrate:**
```bash
# copy BOTH files into the backend (names must match exactly)
cp pdp_yolov8n_seg.onnx pdp_yolov8n_seg.pt  backend/models/
echo "models/*.onnx" >> backend/.gitignore
cd backend && python scripts/prefetch_models.py        # "[ok] found: pdp_yolov8n_seg.onnx, pdp_yolov8n_seg.pt"
uvicorn app.main:app --reload
curl localhost:8000/health                             # "pdp": "yolov8n-seg loaded"
python scripts/run_on_image.py some_photo.jpg          # stage log shows  pdp_segmentation  method=yolov8n_seg
python scripts/evaluate.py --images data/eval --labels data/eval/labels.json --mode both   # yolo-seg used = N/N
```
- The loader prefers `.onnx`, falls back to `.pt`. If ONNX misbehaves, **delete the `.onnx`** and keep the `.pt`.
- Docker: `docker-compose.yml` already mounts `./backend/models` into the container — just restart the backend.
- **Rollback**: delete both files → automatic contour fallback; nothing else changes.
- **Iterate**: run `run_on_image.py` on your worst failures, annotate them, add them to `real/`, rebuild, retrain. 100 well-chosen failure cases beat 300 random extra photos.

---

## Part 4 — Integrate the pretrained models

### 4.1 PaddleOCR — nothing to train
```bash
cd backend && python scripts/prefetch_models.py     # downloads weights the first time; "[ok] loaded"
```
For Docker, add `RUN python scripts/prefetch_models.py` after `COPY . .` in `backend/Dockerfile` to bake weights into the image
(or keep the `paddleocr_cache` volume already in `docker-compose.yml`).
Fine-tune OCR only if `evaluate.py` shows *character-level* errors on crops that are clearly legible to you.

### 4.2 Qwen2.5-VL via Ollama — nothing to train
1. **Host** (pick one):
   - GPU machine / your own box: install Ollama, `ollama pull qwen2.5vl:3b` (`:7b` reads small print better, needs ~8 GB VRAM).
   - Free Colab T4: the snippet in `MIGRATION.md` §3. URL changes every session.
   - CPU-only laptop: works, but expect minutes per label; raise `VLM_TIMEOUT_SECONDS`.
2. **Point the backend at it** — `backend/.env`:
   ```
   ENABLE_VLM=true
   OLLAMA_URL=http://localhost:11434        # or https://xxxx.trycloudflare.com
   VLM_MODEL_NAME=qwen2.5vl:3b
   VLM_TIMEOUT_SECONDS=120
   ```
3. **Verify** — `python scripts/check_vlm.py`. It checks reachability, that the model is pulled, then reads a rendered test label and prints the fields and latency.
4. **Confirm in the pipeline** — `run_on_image.py` stage log: `vlm_wait ... vlm=ok`. If it says `vlm=unavailable -> rule-based fallback` the scan still works, on rules only.
5. **Measure the gain** — `evaluate.py --mode both`. If VLM accuracy isn't clearly above rules, try `qwen2.5vl:7b` before changing anything else.

### 4.3 zxing-cpp — nothing to do
Verify with `run_on_image.py` on a photo with a barcode: `barcode_raw` then `calibration_transfer  raw X -> output Y px/mm`, and the saved `debug_out.jpg` shows a blue box on the barcode.

---

## Part 5 — Deployment note
Backend + ML models need a container host with ≥ 2 GB RAM (Render/Railway/Fly). The VLM is a **separate** service: point `OLLAMA_URL` at a GPU host.
Free tiers can't run it. For a demo, a Colab tunnel or your own machine works; the app degrades to rules if it's unreachable.

## Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| `/health` → `ocr: FAILED` | PaddleOCR 3.x got installed. `pip install paddlepaddle==2.6.2 paddleocr==2.10.0`. |
| `pdp_detection_method` stays `contour_fallback` | Files not named `pdp_yolov8n_seg.onnx/.pt`, not in `backend/models/`, or `ultralytics` import failed (check server log). |
| YOLO mask picks the wrong thing | Dataset has other classes/boxes: `05_check_dataset.py`, retrain single-class. |
| mAP high in Colab, poor on real scans | Synthetic images were in validation, or same product in train and valid. Rebuild with `--group-by-prefix`, real-only valid. |
| `calibration found: false` often | Barcode too small/blurred in the photo: shoot closer or at full resolution; check `evaluate.py` "barcode calibrated" rate. |
| Fonts always FAIL/PASS oddly | Verify `MIN_FONT_SIZE_MM_BY_AREA` in `types.py` against the Rules, and check the blue barcode box in `debug_out.jpg`. |
| `check_vlm.py` 403 through tunnel | Start cloudflared with `--http-host-header localhost:11434`. |
| Scans take ages | VLM on CPU. Move it to a T4, or `ENABLE_VLM=false` for rules-only. |

## Tested here vs. not
Tested in my sandbox: the data tools end to end (photo prep, LabelMe conversion, synthetic generation, split/build, validation and overlay).
Not tested (needs network/GPU): the Colab training run, Ollama/Qwen calls, PaddleOCR download. Those scripts are written defensively and print
exact failure hints — if one breaks, paste me its output.
