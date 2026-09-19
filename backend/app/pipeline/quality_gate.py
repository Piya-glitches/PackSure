"""
LAYER 0 -- IMAGE QUALITY GATE

Laplacian variance (blur) + brightness, now run on the DEWARPED label before
CLAHE (see compliance_engine.py).

Two changes vs. the prototype:
  * "Overexposed" used to be `mean > 235`. A white-background label with
    sparse black print has mean ~245 and is perfectly readable -- it was being
    flagged as glare. Now it also requires LOW CONTRAST (std < 25), i.e. the
    print is actually washed out.
  * Blur threshold lowered 60 -> 40 (env BLUR_VARIANCE_THRESHOLD). The label
    is measured after being resampled to a fixed 1200px width, which smooths
    small crops; 60 sat on the edge of failing a perfectly good 480px sample.
    Tune it on your own photos: scripts/evaluate.py prints these numbers.
"""

import os

import cv2
import numpy as np

from app.pipeline.types import QualityGateResult

BLUR_VARIANCE_THRESHOLD = float(os.environ.get("BLUR_VARIANCE_THRESHOLD", "40"))
MIN_BRIGHTNESS = 25.0
MAX_BRIGHTNESS = 235.0
WASHED_OUT_STD = 25.0


def run_quality_gate(image_bgr: np.ndarray) -> QualityGateResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())

    reasons = []
    if laplacian_variance < BLUR_VARIANCE_THRESHOLD:
        reasons.append(
            f"Image is too blurry (edge variance {laplacian_variance:.1f}, need >= "
            f"{BLUR_VARIANCE_THRESHOLD:.0f}). Hold the camera steady and ensure the label is in focus."
        )
    if brightness_mean < MIN_BRIGHTNESS:
        reasons.append(f"Image is too dark (mean brightness {brightness_mean:.0f}/255). Retake in better lighting.")
    if brightness_mean > MAX_BRIGHTNESS and brightness_std < WASHED_OUT_STD:
        reasons.append(
            f"Image is overexposed/glared (mean brightness {brightness_mean:.0f}/255, contrast {brightness_std:.0f}). "
            "Avoid direct glare on foil/glossy labels."
        )

    return QualityGateResult(
        passed=len(reasons) == 0,
        laplacian_variance=laplacian_variance,
        brightness_mean=brightness_mean,
        reasons=reasons,
    )
