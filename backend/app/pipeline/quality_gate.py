"""
LAYER 0 -- IMAGE QUALITY GATE

Converts the frame to grayscale, convolves with a Laplacian kernel, and
takes the variance of the response. A sharp image has high-frequency edges
everywhere -> high variance. A blurry image has smoothed-out edges -> low
variance. This is a real statistical blur measure used in production
document-scanning pipelines, run here via OpenCV exactly as specified
(Layer 0 of the architecture).

Failing fast here saves Layers 1-4 from wasting compute on an image
nothing could read anyway -- "blurry phone photo" is the #1 real-world
failure mode a field officer or consumer will hit.
"""

import cv2
import numpy as np

from app.pipeline.types import QualityGateResult

BLUR_VARIANCE_THRESHOLD = 60.0
MIN_BRIGHTNESS = 25.0
MAX_BRIGHTNESS = 235.0


def run_quality_gate(image_bgr: np.ndarray) -> QualityGateResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = float(laplacian.var())

    brightness_mean = float(gray.mean())

    reasons = []
    if laplacian_variance < BLUR_VARIANCE_THRESHOLD:
        reasons.append(
            f"Image is too blurry (edge variance {laplacian_variance:.1f}, need >= "
            f"{BLUR_VARIANCE_THRESHOLD}). Hold the camera steady and ensure the label is in focus."
        )
    if brightness_mean < MIN_BRIGHTNESS:
        reasons.append(f"Image is too dark (mean brightness {brightness_mean:.0f}/255). Retake in better lighting.")
    if brightness_mean > MAX_BRIGHTNESS:
        reasons.append(
            f"Image is overexposed/glared (mean brightness {brightness_mean:.0f}/255). "
            "Avoid direct glare on foil/glossy labels."
        )

    return QualityGateResult(
        passed=len(reasons) == 0,
        laplacian_variance=laplacian_variance,
        brightness_mean=brightness_mean,
        reasons=reasons,
    )
