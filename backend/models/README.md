# Fine-tuned model weights go here

- `pdp_yolov8.pt` — YOLOv8 checkpoint fine-tuned to detect the Principal
  Display Panel region on package photos. Until this file is present,
  `app/pipeline/pdp_detection.py` automatically uses a classical
  contour-based fallback (see that file's docstring for why this is
  surfaced honestly rather than silently faked).

- `distilbert-lmpc-ner/` — a DistilBERT/IndicBERT checkpoint fine-tuned
  on the 8-class LMPC field schema (MFR_ADDRESS, NET_QTY, MRP, MFG_DATE,
  CONSUMER_CARE, COUNTRY_OF_ORIGIN, UNIT_PRICE, COMMON_NAME). Until this
  exists, `app/pipeline/field_classifier.py` uses the rule-based
  classifier as the primary decision-driver, with the generic pretrained
  DistilBERT backbone contributing only a supplementary confidence signal.

Training data collection and fine-tuning for both are the concrete Phase 2
roadmap items referenced throughout the pipeline code comments.
