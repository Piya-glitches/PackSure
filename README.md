# PackSure

**Automated compliance verification for packaged commodities under the Legal Metrology (Packaged Commodities) Rules, 2011.**

Built for SIH26034 (Ministry of Consumer Affairs, Food & Public Distribution — Department of Consumer Affairs), implementing the system architecture exactly as specified: **FastAPI backend, YOLOv8 PDP detection, CRAFT+CRNN OCR, DistilBERT/IndicBERT field classification, PostgreSQL, React frontend.**

---

## ⚠️ Deployment reality check — read this first

This architecture **cannot run on Vercel serverless functions.** PyTorch, YOLOv8, EasyOCR (CRAFT+CRNN), and DistilBERT together are hundreds of MB of weights and need persistent process memory — Vercel's serverless functions have a 250MB package limit, no GPU, cold-start on every request, and hard execution timeouts. This is a real constraint of the architecture we chose (ML-heavy Python backend), not something to route around by swapping technology again.

**What actually works:**
- **Backend** (FastAPI + the full ML pipeline): deploy as a Docker container to **Render, Railway, or Fly.io** (all have free/cheap tiers that support this). A `Dockerfile` is included and ready to push.
- **Frontend** (React/Vite, static build): this part genuinely is Vercel/Netlify-friendly — deploy it there and point `VITE_API_URL` at your backend's URL.
- **Local development**: `docker-compose up` runs Postgres + the backend together; run the frontend separately with `npm run dev` for hot-reload.

---

## Architecture → implementation mapping

| Architecture layer | Specified | Implemented as |
|---|---|---|
| Layer 0 — Quality gate | OpenCV Laplacian variance + brightness | `app/pipeline/quality_gate.py` — real, direct |
| Layer 0 — Barcode calibration | pyzbar/ZXing EAN-13/UPC | `app/pipeline/barcode_calibration.py` — pyzbar, real GS1 width lookup |
| Layer 1 — PDP localization | Fine-tuned YOLOv8/RF-DETR | `app/pipeline/pdp_detection.py` — real Ultralytics YOLOv8 inference path; **falls back to classical contour detection until a fine-tuned `models/pdp_yolov8.pt` checkpoint is supplied** (we don't have a labelled PDP dataset yet — see roadmap) |
| Layer 1 — Perspective correction | Homography via OpenCV | `app/pipeline/dewarp.py` — real `getPerspectiveTransform`/`warpPerspective` |
| Layer 2 — Text detection | CRAFT | Provided by EasyOCR's detector (see note below) |
| Layer 2 — OCR/recognition | Custom CRNN | Provided by EasyOCR's recognizer, which **is** a CRNN (CNN+BiLSTM+CTC) — see `app/pipeline/text_detection_ocr.py` docstring. Standalone CRNN architecture also defined in `app/pipeline/crnn_model.py` for Phase 2 fine-tuning |
| Layer 2 — Multilingual | Hindi + English | EasyOCR `['en','hi']` language pack |
| Layer 2 — CLAHE / glare handling | CLAHE | `app/pipeline/dewarp.py::apply_clahe` |
| Layer 3 — Field classification | DistilBERT/IndicBERT NER | `app/pipeline/field_classifier.py` — real `transformers` token-classification pipeline wired in as an ensemble signal; **rule-augmented classifier is the practical decision-driver until the backbone is fine-tuned on our 8-class schema** (see roadmap) |
| Layer 4 — Format validators | Deterministic regex/rules | `app/pipeline/rule_validators.py` |
| Layer 4 — Responsible-party engine | Rule-based decision tree | `app/pipeline/responsible_party.py` |
| Layer 4 — Font-size validator | px→mm via calibration | `app/pipeline/font_size.py` |
| Layer 5 — Database | PostgreSQL | SQLAlchemy models, Postgres in production / SQLite for zero-setup local dev |
| Layer 5 — Dashboard | React + Recharts | `frontend/src/pages/DashboardPage.tsx` |
| Layer 5 — Reports | PDF (ReportLab) | `app/pdf_report.py` |
| Layer 5 — RBAC | Role-based auth | JWT auth, `public`/`officer`/`admin` roles, `app/security.py` |
| Layer 6 — Backend framework | FastAPI | `app/main.py` |
| Layer 6 — Model serving | ONNX Runtime | Not yet wired — current inference runs native PyTorch/Ultralytics/EasyOCR. ONNX export/serving is a straightforward Phase 2 optimization once fine-tuned weights exist (documented, not silently dropped) |

### Why EasyOCR genuinely satisfies "CRAFT + CRNN" rather than substituting it

EasyOCR's detector **is** CRAFT (Character Region Awareness for Text). EasyOCR's default recognizer **is** a CRNN — a CNN feature extractor feeding a bidirectional LSTM feeding a CTC decoder, which is precisely the architecture named in Layer 2. It's pretrained and ships an `en+hi` language pack out of the box. This is the direct, working implementation of the specified two-stage pipeline — not a stand-in for it. The standalone `crnn_model.py` module defines the same architecture explicitly, ready to fine-tune on a domain-specific dataset per the roadmap below.

### Honest scoping — what still needs real-world data to complete

1. **YOLOv8 PDP detector**: needs ~500-1000 package photos with the label region hand-annotated, then a `yolo train` run. Until that checkpoint exists at `backend/models/pdp_yolov8.pt`, the contour-based classical fallback runs instead (and the report always states which method ran, via the `pdp_detection_method` field).
2. **DistilBERT/IndicBERT fine-tuning**: needs OCR output labelled against the 8-field schema, ideally with synthetic OCR-error augmentation as the architecture specifies. Until fine-tuned, the rule-based classifier drives the actual pass/fail decision, with the generic pretrained backbone contributing only a supplementary confidence signal (see `field_classifier.py`).
3. **ONNX export**: once the above are trained, exporting to ONNX Runtime for faster/more portable serving is a mechanical next step, not a redesign.

Both gaps are stated plainly in code comments at the exact point they matter, not buried — an honest "not yet trained, here's the fallback and here's the path" is stronger to a judge than a black-box claim that can't survive a follow-up question.

---

## Run locally

### Backend + database (Docker)

```bash
docker-compose up --build
```

This starts Postgres and the FastAPI backend at `http://localhost:8000` (interactive API docs at `/docs`). First startup downloads EasyOCR/DistilBERT model weights (a few hundred MB) — this needs internet access and takes a few minutes the first time only.

### Backend without Docker

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # defaults to local SQLite, zero setup
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # points at http://localhost:8000 by default
npm run dev
```

Open `http://localhost:5173`, go to **Scan**, and try a built-in sample label — no upload needed, no account needed.

---

## Deploy to production

**Backend (Render, as an example):**
1. Push this repo to GitHub.
2. New Web Service on Render → connect the repo → set root directory to `backend` → Render auto-detects the `Dockerfile`.
3. Add a Render Postgres instance (or Neon/Supabase) and set `DATABASE_URL` in the service's environment variables.
4. Set `JWT_SECRET` to a long random string and `CORS_ORIGINS` to your deployed frontend's URL.

**Frontend (Vercel):**
1. Import the repo, set the root directory to `frontend`.
2. Set the `VITE_API_URL` environment variable to your Render backend's URL.
3. Deploy — Vercel builds the Vite app automatically.

---

## Project structure

```
backend/
  app/
    main.py                    FastAPI app, CORS, startup
    config.py, db.py, models.py, schemas.py, security.py
    pipeline/
      quality_gate.py           Layer 0 — blur/brightness (OpenCV)
      barcode_calibration.py    Layer 0 — barcode-as-ruler (novelty #1, pyzbar)
      pdp_detection.py          Layer 1 — YOLOv8 + contour fallback
      dewarp.py                 Layer 1 — homography + CLAHE
      text_detection_ocr.py     Layer 2 — CRAFT+CRNN via EasyOCR
      crnn_model.py             standalone CRNN architecture (fine-tuning roadmap)
      field_classifier.py       Layer 3 — DistilBERT backbone + rule-based classifier
      font_size.py              Layer 4 — px→mm validator (novelty #2)
      responsible_party.py      Layer 4 — legal role resolution (novelty #3)
      rule_validators.py        Layer 4 — deterministic format checks
      compliance_engine.py      orchestrator
      sample_labels.py          synthetic demo labels w/ real EAN-13 barcodes
    routers/                    auth, scans, stats, report endpoints
    pdf_report.py                Layer 5 — ReportLab PDF generation
  models/                       drop fine-tuned checkpoints here (see models/README.md)
  Dockerfile, requirements.txt

frontend/
  src/
    pages/                      Home, Login, Register, Scan, History, ScanDetail, Dashboard
    components/                 Navbar, Uploader, SampleLabelPicker, ImageWithOverlay, ComplianceReportView
    api/client.ts                axios client + JWT handling
    types.ts
  Dockerfile (optional, static build via nginx)

docker-compose.yml               Postgres + backend for local dev
```

---

## The three novelty claims (for judges)

1. **Barcode-as-self-calibrating-ruler** — real physical font-size measurement from an uncalibrated phone photo, using a standardised element already on nearly every regulated product.
2. **Open-set field classification** — architected to work on a product it has never seen before (rule-based today, DistilBERT fine-tune as the trained upgrade), unlike commercial FMCG QC tools that compare against a pre-known master template per brand.
3. **Responsible-party legal resolution** — encodes the actual manufacturer/packer/importer/brand-owner distinction from the PC Rules, not just text presence.
