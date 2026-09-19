"""
End-to-end smoke test on a REAL photo (systechchange.pdf Step 3).

Run from the backend/ directory (NOT from scripts/):

    python scripts/run_on_image.py path/to/amul_butter.jpg
    python scripts/run_on_image.py photo.jpg --no-vlm          # rules-only
    python scripts/run_on_image.py photo.jpg --out debug.jpg   # save annotated image

What to look for:
  * barcode_raw / calibration_transfer  -> did zxing find the barcode and what px/mm did it produce?
  * pdp_segmentation method             -> yolov8n_seg or contour_fallback
  * dewarp method                       -> homography or cylindrical
  * vlm_wait note                       -> vlm=ok or fallback
  * per-field text, status, measured mm
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.config  # noqa: F401,E402  (loads .env before the pipeline reads os.environ)
import cv2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--no-vlm", action="store_true")
ap.add_argument("--out", default="debug_out.jpg")
args = ap.parse_args()
if args.no_vlm:
    os.environ["ENABLE_VLM"] = "false"

from app.pipeline.compliance_engine import run_full_pipeline, warm_up  # noqa: E402

img = cv2.imread(args.image)
if img is None:
    sys.exit(f"Could not read {args.image}")

print("warm-up:", warm_up())
report, processed = run_full_pipeline(img)

print("\n== STAGES ==")
for s in report.processing_log:
    print(f"  {s.stage:24s} {s.duration_ms:8.0f} ms  {s.note}")

c = report.calibration
print("\n== CALIBRATION ==")
print("  found" if c.found else "  NOT FOUND", c.symbology, c.raw_value or "", f"{c.px_per_mm:.2f} px/mm (dewarped space)" if c.px_per_mm else "")
q = report.quality_gate
print(f"== QUALITY == passed={q.passed} blur_var={q.laplacian_variance:.1f} brightness={q.brightness_mean:.0f} {q.reasons}")

print("\n== FIELDS ==")
for f in report.fields:
    mm = f"{f.font_height_mm:.2f}mm/{f.min_required_mm:.1f}mm" if f.font_height_mm else "-"
    print(f"  {f.field_key:18s} {f.status:9s} conf={f.confidence:.2f} font={mm:14s} {f.extracted_text!r}")

print(f"\n== RESPONSIBLE PARTY == {report.responsible_party.role}: {report.responsible_party.reasoning}")
print(f"\n== VERDICT == {report.overall_status}  score={report.compliance_score}")
for v in report.violations:
    print(f"  [{v.severity}] {v.rule_code}: {v.description[:110]}")

for f in report.fields:
    if f.bbox:
        b = f.bbox
        cv2.rectangle(processed, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), (0, 160, 0), 2)
        cv2.putText(processed, f.field_key, (int(b.x), max(int(b.y) - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 0), 1)
if c.found and c.bbox:
    b = c.bbox
    cv2.rectangle(processed, (int(b.x), int(b.y)), (int(b.x + b.w), int(b.y + b.h)), (200, 100, 0), 2)
cv2.imwrite(args.out, processed)
print(f"\nAnnotated dewarped image -> {args.out}  (check the barcode box lands on the barcode)")
