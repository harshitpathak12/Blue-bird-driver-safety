"""
YOLO card detection + Qwen OCR + Indian DL rules — single-frame validation.

Weights: ``models/driving_license/best.pt`` (override with DL_YOLO_WEIGHTS).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.logger import get_logger

from .image_utils import crop_bbox
from .license_rules import validate_indian_dl
from .qwen_ocr import is_placeholder_ocr_text, qwen_ocr_and_validate

logger = get_logger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent
# driving_license_model → scripts → training → repo root
_PROJECT_ROOT = _PKG_ROOT.parent.parent.parent
_DEFAULT_WEIGHTS = _PROJECT_ROOT / "models" / "driving_license" / "best.pt"
DEFAULT_WEIGHTS = Path(os.environ.get("DL_YOLO_WEIGHTS", str(_DEFAULT_WEIGHTS)))

IMGSZ = int(os.environ.get("DL_IMGSZ", "416"))
DETECTION_CONF_THRESHOLD = float(os.environ.get("DL_DET_CONF", "0.5"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE == "cuda"

_yolo_model: YOLO | None = None


def _warmup_yolo(model: YOLO, runs: int = 2) -> None:
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    for _ in range(runs):
        model.predict(dummy, imgsz=IMGSZ, conf=0.25, verbose=False, half=USE_HALF, device=DEVICE)
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def get_yolo(weights_path: Path | None = None) -> YOLO:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    wp = Path(weights_path or DEFAULT_WEIGHTS)
    if not wp.is_file():
        raise FileNotFoundError(
            f"Driving licence YOLO weights not found: {wp}. "
            "Place best.pt under models/driving_license/ or set DL_YOLO_WEIGHTS.",
        )
    logger.info("Loading DL YOLO from %s (%s)", wp, DEVICE)
    _yolo_model = YOLO(str(wp))
    try:
        _yolo_model.model.float()
    except Exception:
        pass
    _yolo_model.fuse()
    _yolo_model.to(DEVICE)
    if USE_HALF:
        _yolo_model.model.half()
    _warmup_yolo(_yolo_model)
    return _yolo_model


def validate_license_frame(frame: np.ndarray, model: YOLO | None = None) -> dict[str, Any]:
    """
    Full pipeline on one BGR frame. Returns dict with verdict, dl_numbers, ocr_text, rule fields.
    """
    t_total = time.perf_counter()
    if frame is None or frame.size == 0:
        return {"verdict": "error", "message": "Empty frame"}

    m = model or get_yolo()

    t0 = time.perf_counter()
    with torch.inference_mode():
        results = m.predict(
            frame,
            imgsz=IMGSZ,
            conf=DETECTION_CONF_THRESHOLD,
            verbose=False,
            device=DEVICE,
            half=USE_HALF,
        )
    yolo_ms = (time.perf_counter() - t0) * 1000

    if not results or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return {
            "verdict": "unknown",
            "message": "No driving licence detected in frame",
            "timings": {"yolo_ms": round(yolo_ms)},
        }

    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    bbox = boxes.xyxy[best_idx].detach().float().cpu().numpy().tolist()
    det_conf = float(boxes.conf[best_idx])

    crop = crop_bbox(frame, bbox, padding=10)
    if crop is None:
        return {
            "verdict": "unknown",
            "message": "Failed to crop detected region",
            "detection_confidence": det_conf,
            "timings": {"yolo_ms": round(yolo_ms)},
        }

    ch, cw = crop.shape[:2]
    if max(ch, cw) < 400:
        scale = 400 / max(ch, cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)

    t0 = time.perf_counter()
    ocr_result = qwen_ocr_and_validate(crop)
    ocr_ms = (time.perf_counter() - t0) * 1000

    ocr_text = ocr_result.text or ""
    if not ocr_text.strip() or is_placeholder_ocr_text(ocr_text):
        return {
            "verdict": "no_text",
            "message": "Text not visible on card",
            "detection_confidence": det_conf,
            "bbox": bbox,
            "ocr_text": "",
            "dl_numbers": [],
            "validity_end": None,
            "timings": {
                "yolo_ms": round(yolo_ms),
                "ocr_ms": round(ocr_ms),
                "total_ms": round((time.perf_counter() - t_total) * 1000),
            },
        }

    rule = validate_indian_dl(ocr_text)
    label = rule.get("label", "unknown")
    total_ms = (time.perf_counter() - t_total) * 1000

    return {
        "verdict": label,
        "message": rule.get("reason", ""),
        "detection_confidence": det_conf,
        "bbox": bbox,
        "ocr_text": ocr_text,
        "qwen_verdict": ocr_result.is_valid,
        "qwen_confidence": ocr_result.confidence,
        "qwen_reason": ocr_result.reason,
        "rule_label": label,
        "rule_confidence": rule.get("confidence", 0.0),
        "rule_reason": rule.get("reason", ""),
        "dl_numbers": rule.get("dl_numbers", []),
        "validity_end": rule.get("validity_end"),
        "timings": {
            "yolo_ms": round(yolo_ms),
            "ocr_ms": round(ocr_ms),
            "total_ms": round(total_ms),
        },
    }
