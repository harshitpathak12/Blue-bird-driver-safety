"""
Per-connection DL stream session: YOLO every frame, Qwen OCR in a background thread
with throttling and caching — same model as ``driver_license_detection`` ``/stream``.

Each WebSocket connection must use its own ``DlStreamSession`` instance (isolated OCR thread).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .dl_validator import DETECTION_CONF_THRESHOLD, DEVICE, IMGSZ, USE_HALF
from .image_utils import crop_bbox
from .license_rules import validate_indian_dl
from utils.logger import get_logger

from .qwen_ocr import is_placeholder_ocr_text, qwen_ocr_and_validate
from .visualization import draw_bbox, draw_validation_status

_log = get_logger(__name__)

# Match driver_license_detection/api/websocket.py tuning (env-overridable).
OCR_RESUBMIT_SEC = float(os.environ.get("OCR_RESUBMIT_SEC", "3.0"))
BBOX_CHANGE_THRESHOLD = float(os.environ.get("BBOX_CHANGE_THRESHOLD", "0.15"))
NO_CARD_CLEAR_FRAMES = int(os.environ.get("NO_CARD_CLEAR_FRAMES", "5"))


class DlStreamSession:
    """
    Stateful stream processor for one client: fast YOLO path + async OCR worker.
    """

    def __init__(
        self,
        stream_width: int,
        stream_height: int,
        encode_params: list[int],
    ) -> None:
        self.stream_width = stream_width
        self.stream_height = stream_height
        self._encode_params = encode_params

        self._ocr_lock = threading.Lock()
        self._ocr_crop: np.ndarray | None = None
        self._ocr_pending = False
        self._ocr_busy = False
        self._ocr_result_text: str = ""
        self._ocr_result_rule: dict[str, Any] | None = None
        self._ocr_thread: threading.Thread | None = None
        self._ocr_stop = threading.Event()

        self._last_bbox_center: tuple[float, float] | None = None
        self._no_card_counter = 0
        self._last_ocr_submit_time = 0.0

    def start(self) -> None:
        self._ensure_ocr_thread()

    def stop(self) -> None:
        self._ocr_stop.set()
        if self._ocr_thread is not None and self._ocr_thread.is_alive():
            self._ocr_thread.join(timeout=3.0)

    def _clear_ocr_cache(self) -> None:
        with self._ocr_lock:
            self._ocr_result_text = ""
            self._ocr_result_rule = None
        self._last_bbox_center = None

    def _ocr_worker(self) -> None:
        while not self._ocr_stop.is_set():
            crop = None
            with self._ocr_lock:
                if self._ocr_pending and self._ocr_crop is not None:
                    crop = self._ocr_crop
                    self._ocr_pending = False

            if crop is None:
                self._ocr_stop.wait(0.01)
                continue

            with self._ocr_lock:
                self._ocr_busy = True
            try:
                result = qwen_ocr_and_validate(crop)
                raw = (result.text or "").strip()
                if is_placeholder_ocr_text(raw):
                    text = ""
                    rule = {
                        "label": "unknown",
                        "confidence": 0.0,
                        "reason": "OCR did not return readable text (retry)",
                        "dl_numbers": [],
                    }
                else:
                    text = raw
                    rule = validate_indian_dl(text)
                    if os.environ.get("DL_DEBUG_OCR", "").strip().lower() in ("1", "true", "yes", "on"):
                        _log.info(
                            "DL_DEBUG_OCR qwen is_valid=%s text_len=%s rule_label=%s reason=%s",
                            result.is_valid,
                            len(text),
                            rule.get("label"),
                            (rule.get("reason") or "")[:160],
                        )
                with self._ocr_lock:
                    self._ocr_result_text = text
                    self._ocr_result_rule = rule
            except Exception:
                pass
            finally:
                with self._ocr_lock:
                    self._ocr_busy = False

    def _ensure_ocr_thread(self) -> None:
        if self._ocr_thread is None or not self._ocr_thread.is_alive():
            self._ocr_stop.clear()
            self._ocr_thread = threading.Thread(
                target=self._ocr_worker,
                daemon=True,
                name="dl_stream_qwen_ocr",
            )
            self._ocr_thread.start()

    def _submit_ocr_crop(self, crop: np.ndarray) -> None:
        with self._ocr_lock:
            self._ocr_crop = crop
            self._ocr_pending = True
        self._last_ocr_submit_time = time.time()

    def _get_ocr_result(self) -> tuple[str, dict[str, Any] | None]:
        with self._ocr_lock:
            return self._ocr_result_text, self._ocr_result_rule

    def _get_ocr_flags(self) -> tuple[bool, bool]:
        with self._ocr_lock:
            return self._ocr_pending, self._ocr_busy

    def _bbox_center(self, bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _bbox_changed(self, new_center: tuple[float, float]) -> bool:
        if self._last_bbox_center is None:
            return True
        dx = abs(new_center[0] - self._last_bbox_center[0]) / float(self.stream_width)
        dy = abs(new_center[1] - self._last_bbox_center[1]) / float(self.stream_height)
        return max(dx, dy) > BBOX_CHANGE_THRESHOLD

    def fast_detect(
        self,
        frame: np.ndarray,
        model: YOLO,
        qwen_enabled: bool,
    ) -> tuple[list[dict[str, Any]], bytes | None]:
        with torch.inference_mode():
            results = model.predict(
                frame,
                imgsz=IMGSZ,
                conf=DETECTION_CONF_THRESHOLD,
                verbose=False,
                device=DEVICE,
                half=USE_HALF,
            )

        out_img = frame.copy()
        detections: list[dict[str, Any]] = []

        has_card = False
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.detach().float().cpu().numpy()
            xyxys = boxes.xyxy.detach().float().cpu().numpy()
            indices = np.argsort(-confs)

            for rank, idx in enumerate(indices.tolist()):
                conf = float(confs[idx])
                if conf < DETECTION_CONF_THRESHOLD:
                    continue
                bbox = xyxys[idx].tolist()
                has_card = True

                det: dict[str, Any] = {
                    "bbox": bbox,
                    "confidence": conf,
                    "class": "driving_license",
                    "ocr_lines": [],
                    "ocr_text": "",
                    "validation_label": "unknown",
                    "validation_confidence": 0.0,
                    "validation_reason": "",
                    "dl_numbers": [],
                    "validity_end": None,
                }

                crop = crop_bbox(frame, bbox, padding=10)
                if crop is not None and qwen_enabled and rank == 0:
                    center = self._bbox_center(bbox)
                    card_changed = self._bbox_changed(center)

                    if card_changed:
                        self._clear_ocr_cache()
                    self._last_bbox_center = center
                    self._no_card_counter = 0

                    ch, cw = crop.shape[:2]
                    if max(ch, cw) < 400:
                        scale = 400 / max(ch, cw)
                        crop = cv2.resize(
                            crop,
                            (int(cw * scale), int(ch * scale)),
                            interpolation=cv2.INTER_CUBIC,
                        )

                    ocr_pending, ocr_busy = self._get_ocr_flags()
                    time_since_submit = time.time() - self._last_ocr_submit_time

                    should_submit = (
                        (not ocr_busy and not ocr_pending)
                        and (card_changed or time_since_submit >= OCR_RESUBMIT_SEC)
                    )
                    if should_submit:
                        self._submit_ocr_crop(crop)

                    ocr_text, ocr_rule = self._get_ocr_result()

                    if ocr_busy or ocr_pending:
                        if ocr_text and not is_placeholder_ocr_text(ocr_text) and ocr_rule:
                            det["ocr_text"] = ocr_text
                            det["ocr_lines"] = [{"text": ocr_text, "confidence": 1.0}]
                            lbl = ocr_rule.get("label", "processing")
                            if lbl == "invalid":
                                lbl = "processing"
                                det["validation_reason"] = "OCR in progress…"
                                det["validation_confidence"] = 0.0
                            else:
                                det["validation_reason"] = ocr_rule.get("reason", "OCR updating…")
                                det["validation_confidence"] = ocr_rule.get("confidence", 0.0)
                            det["validation_label"] = lbl
                            det["dl_numbers"] = ocr_rule.get("dl_numbers", [])
                            det["validity_end"] = ocr_rule.get("validity_end")
                        else:
                            det["validation_label"] = "processing"
                            det["validation_confidence"] = 0.0
                            det["validation_reason"] = "OCR in progress…"
                    elif ocr_text and not is_placeholder_ocr_text(ocr_text):
                        det["ocr_text"] = ocr_text
                        det["ocr_lines"] = [{"text": ocr_text, "confidence": 1.0}]
                        if ocr_rule:
                            det["validation_label"] = ocr_rule.get("label", "unknown")
                            det["validation_confidence"] = ocr_rule.get("confidence", 0.0)
                            det["validation_reason"] = ocr_rule.get("reason", "")
                            det["dl_numbers"] = ocr_rule.get("dl_numbers", [])
                            det["validity_end"] = ocr_rule.get("validity_end")
                    else:
                        if ocr_rule is not None and not (ocr_text or "").strip():
                            det["validation_label"] = "no_text"
                            det["validation_confidence"] = 0.0
                            det["validation_reason"] = "Text not visible on card"
                            det["dl_numbers"] = []
                            det["validity_end"] = None
                        else:
                            det["validation_label"] = "unknown"
                            det["validation_reason"] = "Waiting for OCR…"

                detections.append(det)

                vl = det.get("validation_label", "unknown")
                dl_nums = det.get("dl_numbers") or []
                if vl == "processing":
                    label_text = "OCR…"
                elif vl == "no_text":
                    label_text = "NO TEXT"
                elif vl == "valid" and dl_nums:
                    label_text = f"VALID | {str(dl_nums[0])[:22]}"
                elif vl == "invalid" and dl_nums:
                    label_text = f"INVALID | {str(dl_nums[0])[:20]}"
                elif vl == "valid":
                    label_text = "VALID"
                elif vl == "invalid":
                    label_text = "INVALID"
                elif dl_nums:
                    label_text = str(dl_nums[0])[:24]
                else:
                    label_text = "License"
                color = (
                    (0, 255, 0) if vl == "valid"
                    else (0, 0, 255) if vl == "invalid"
                    else (0, 255, 255) if vl == "processing"
                    else (0, 165, 255) if vl == "no_text"
                    else (0, 200, 255)
                )
                draw_bbox(out_img, bbox, label=label_text, confidence=conf, color=color)

        if not has_card:
            self._no_card_counter += 1
            if self._no_card_counter >= NO_CARD_CLEAR_FRAMES:
                self._clear_ocr_cache()

        for d in detections:
            vl = d.get("validation_label")
            reason = d.get("validation_reason", "")
            if vl == "processing":
                draw_validation_status(out_img, "processing", reason or "OCR in progress…")
                break
            if vl == "no_text":
                draw_validation_status(out_img, "no_text", reason or "Text not visible on card")
                break
            if vl in ("valid", "invalid"):
                draw_validation_status(out_img, vl, reason)
                break

        ok, buf = cv2.imencode(".jpg", out_img, self._encode_params)
        return detections, buf.tobytes() if ok else None
