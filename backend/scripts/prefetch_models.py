"""
Download / verify every model the backend needs. Run once after install, or during `docker build`
so the container never downloads at request time:

    python scripts/prefetch_models.py

  PaddleOCR (det + rec + angle cls, Hindi pack)  -> auto-downloaded to ~/.paddleocr
  YOLOv8n-seg PDP checkpoint                     -> just checks backend/models/ (you train it)
  Qwen2.5-VL                                     -> served by Ollama; use scripts/check_vlm.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.config  # noqa: F401,E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

print("== PaddleOCR ==")
try:
    from app.pipeline.text_detection_ocr import get_reader
    get_reader()
    print("[ok] loaded (weights cached in ~/.paddleocr)")
except Exception as e:
    print("[FAIL]", e)

print("\n== YOLOv8n-seg PDP checkpoint ==")
found = [f for f in ("pdp_yolov8n_seg.onnx", "pdp_yolov8n_seg.pt") if os.path.exists(os.path.join(MODELS, f))]
print("[ok] found:" if found else "[--] none found in backend/models/ -> contour fallback will be used.", ", ".join(found))
