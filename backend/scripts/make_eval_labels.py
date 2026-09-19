"""
Bootstrap labels.json for scripts/evaluate.py -- you CORRECT it instead of typing everything.

    python scripts/make_eval_labels.py --images data/eval/ --out data/eval/labels.draft.json

Runs the pipeline on every photo and writes what it extracted. Then OPEN THE FILE AND FIX IT while looking
at each photo: correct wrong values, set fields that are not printed to null. Rename to labels.json.

!! If you skip the review, evaluate.py just measures the pipeline against itself (~100 % "accurate").
   The draft is a typing shortcut, not ground truth.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config  # noqa: F401,E402
import cv2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

from app.pipeline.compliance_engine import run_full_pipeline  # noqa: E402

draft = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
files = sorted(f for f in os.listdir(args.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
for i, fname in enumerate(files, 1):
    if fname in draft:
        continue                                   # resume: don't overwrite hand-corrected entries
    img = cv2.imread(os.path.join(args.images, fname))
    if img is None:
        continue
    report, _ = run_full_pipeline(img)
    draft[fname] = {f.field_key: f.extracted_text for f in report.fields}
    json.dump(draft, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[{i}/{len(files)}] {fname}")
print(f"\nDraft written to {args.out}. Now review EVERY value against the photo.")
