"""
Measure the pipeline against your own labelled photos -- this is what tells
you WHAT TO TRAIN NEXT instead of guessing.

    python scripts/evaluate.py --images data/eval/ --labels data/eval/labels.json --mode both

labels.json (null = the field is genuinely not printed):
{
  "amul_butter.jpg": {"MRP": "Rs. 60", "NET_QTY": "100 g", "MFG_DATE": null,
                      "MFR_ADDRESS": "...", "CONSUMER_CARE": "1800-258-3333",
                      "COUNTRY_OF_ORIGIN": "India", "UNIT_PRICE": null, "COMMON_NAME": "Butter"}
}
Only fields you list are scored. Text match = fuzzy ratio >= 0.75 on normalised text.

--mode rules | vlm | both   (both = the comparison that justifies the VLM)

Also prints per-image barcode / PDP / quality numbers: use them to decide whether
you need the YOLO-seg checkpoint (pdp=contour_fallback often?), whether barcodes
are being found (calibration rate), and whether BLUR_VARIANCE_THRESHOLD is sane.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config  # noqa: F401,E402
import cv2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--mode", choices=["rules", "vlm", "both"], default="both")
args = ap.parse_args()

from app.pipeline.compliance_engine import run_full_pipeline  # noqa: E402


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower().replace(".", "")).strip()


def match(pred, truth):
    if truth is None:
        return pred is None
    if pred is None:
        return False
    a, b = norm(pred), norm(truth)
    return b in a or SequenceMatcher(None, a, b).ratio() >= 0.75


labels = json.load(open(args.labels, encoding="utf-8"))
modes = ["rules", "vlm"] if args.mode == "both" else [args.mode]
stats = {m: defaultdict(lambda: [0, 0]) for m in modes}
infra = {"images": 0, "calibrated": 0, "seg": 0, "cyl": 0, "blur_fail": 0}

for fname, truth in labels.items():
    img = cv2.imread(os.path.join(args.images, fname))
    if img is None:
        print("skip (unreadable):", fname)
        continue
    for mode in modes:
        os.environ["ENABLE_VLM"] = "true" if mode == "vlm" else "false"
        report, _ = run_full_pipeline(img)
        got = {f.field_key: f.extracted_text for f in report.fields}
        for key, expected in truth.items():
            ok = match(got.get(key), expected)
            stats[mode][key][0] += int(ok)
            stats[mode][key][1] += 1
        if mode == modes[0]:
            infra["images"] += 1
            infra["calibrated"] += int(report.calibration.found)
            infra["seg"] += int(report.pdp_detection_method == "yolov8n_seg")
            infra["cyl"] += int(any("cylindrical" in s.note for s in report.processing_log))
            infra["blur_fail"] += int(not report.quality_gate.passed)
            print(f"{fname:30s} pdp={report.pdp_detection_method:15s} calib={report.calibration.found!s:5s} "
                  f"blur={report.quality_gate.laplacian_variance:6.1f} status={report.overall_status}")

print("\n=== FIELD ACCURACY (correct / scored) ===")
keys = sorted({k for m in modes for k in stats[m]})
print(f"{'field':20s}" + "".join(f"{m:>14s}" for m in modes))
for k in keys:
    print(f"{k:20s}" + "".join(f"{stats[m][k][0]:>8d}/{stats[m][k][1]:<5d}" for m in modes))
for m in modes:
    c = sum(v[0] for v in stats[m].values()); t = sum(v[1] for v in stats[m].values())
    print(f"overall {m}: {c}/{t} = {100*c/max(t,1):.1f}%")

n = max(infra["images"], 1)
print(f"\n=== INFRASTRUCTURE === images={infra['images']}  barcode calibrated={infra['calibrated']}/{n}  "
      f"yolo-seg used={infra['seg']}/{n}  cylindrical={infra['cyl']}/{n}  quality-gate failures={infra['blur_fail']}/{n}")
