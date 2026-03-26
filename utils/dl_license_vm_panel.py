"""
Side panel UI for driving licence WebSocket verification — matches
`driver_license_detection/vm_client.py` (License Check + Result + FPS).
"""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

PANEL_WIDTH = 360
FONT = cv2.FONT_HERSHEY_SIMPLEX

COL_BG = (28, 28, 28)
COL_WHITE = (240, 240, 240)
COL_LIGHT = (200, 200, 200)
COL_GRAY = (140, 140, 140)
COL_GREEN = (0, 200, 100)
COL_YELLOW = (0, 210, 255)
COL_RED = (0, 70, 255)
COL_CYAN = (220, 180, 0)


def _text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.5, color=COL_WHITE, thick: int = 1) -> None:
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _divider(img: np.ndarray, y: int, w: int) -> None:
    cv2.line(img, (14, y), (w - 14, y), (60, 60, 60), 1)


def _friendly_verdict(label: str) -> tuple[str, tuple[int, int, int]]:
    l = (label or "").lower()
    if l == "valid":
        return "VALID", COL_GREEN
    if l == "invalid":
        return "INVALID", COL_RED
    if l == "no_text":
        return "TEXT NOT VISIBLE", COL_YELLOW
    if l == "unknown":
        return "UNKNOWN", COL_YELLOW
    if l == "processing":
        return "PROCESSING", COL_CYAN
    return (label or "UNKNOWN").upper(), COL_YELLOW


def select_primary_detection(dets: list[Any]) -> dict[str, Any]:
    """Same priority as driver_license_detection/vm_client.py (no_text → processing → valid/invalid → dl → first)."""

    def _has_dl(d: dict[str, Any]) -> bool:
        n = d.get("dl_numbers") or []
        return isinstance(n, list) and len(n) > 0

    if not isinstance(dets, list):
        return {}
    for d in dets:
        if str(d.get("validation_label") or "").lower() == "no_text":
            return d
    for d in dets:
        if str(d.get("validation_label") or "").lower() == "processing":
            return d
    for d in dets:
        vl = str(d.get("validation_label") or "").lower()
        if vl in ("valid", "invalid"):
            return d
    for d in dets:
        if _has_dl(d):
            return d
    if dets:
        return dets[0]
    return {}


def extract_dl_number_for_finalize(d0: dict[str, Any]) -> str:
    """DL number for POST /finalize-dl; mirrors vm_client panel regex fallback."""
    dl_nums = d0.get("dl_numbers") or []
    dl = str(dl_nums[0]).strip() if isinstance(dl_nums, list) and dl_nums else ""
    if not dl and d0.get("ocr_text"):
        ocr = (d0.get("ocr_text") or "").strip()
        for pat in (
            r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?(?:19|20)\d{2}\s*\d{5,7}\b",
            r"\b[A-Z]{2}[\s\-]?\d{2,}[\s\-]?\d{4,}\b",
        ):
            m = re.search(pat, ocr, re.IGNORECASE)
            if m:
                return m.group(0).strip()
    return dl


def build_panel(info: dict[str, Any], fps: float, frame_h: int) -> np.ndarray:
    """Same layout as driver_license_detection/vm_client.build_panel."""
    panel = np.full((frame_h, PANEL_WIDTH, 3), COL_BG, dtype=np.uint8)
    w = PANEL_WIDTH
    x = 18
    y = 34

    _text(panel, "License Check", x, y, 0.62, COL_CYAN, 2)
    y += 12
    _divider(panel, y, w)
    y += 26

    dets = info.get("detections") or []
    if not isinstance(dets, list):
        dets = []

    d0 = select_primary_detection(dets)

    verdict = str(d0.get("validation_label") or "unknown")
    v_text, v_col = _friendly_verdict(verdict)

    _text(panel, "Result", x, y, 0.46, COL_GRAY, 1)
    y += 28
    _text(panel, v_text, x, y, 0.85, v_col, 2)
    y += 36

    if verdict == "no_text":
        _text(panel, "Your text is not visible", x, y, 0.48, COL_LIGHT, 1)
        y += 28
    elif verdict == "valid":
        dl_nums = d0.get("dl_numbers") or []
        dl = str(dl_nums[0]).strip() if isinstance(dl_nums, list) and dl_nums else ""
        if not dl and d0.get("ocr_text"):
            ocr = (d0.get("ocr_text") or "").strip()
            for pat in (
                r"\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?(?:19|20)\d{2}\s*\d{5,7}\b",
                r"\b[A-Z]{2}[\s\-]?\d{2,}[\s\-]?\d{4,}\b",
            ):
                m = re.search(pat, ocr, re.IGNORECASE)
                if m:
                    dl = m.group(0).strip()
                    break
        ve = d0.get("validity_end") or ""
        if dl:
            _text(panel, f"DL: {dl[:28]}", x, y, 0.48, COL_WHITE, 1)
            y += 22
        if ve:
            _text(panel, f"Valid till: {ve}", x, y, 0.45, COL_GREEN, 1)
            y += 22
    elif verdict == "invalid":
        reason = str(d0.get("validation_reason") or "").replace("\n", " ").strip()[:52]
        if reason:
            _text(panel, reason, x, y, 0.44, COL_LIGHT, 1)
            y += 22

    _divider(panel, y, w)
    y += 18

    _text(panel, f"FPS: {fps:.1f}", w - 120, frame_h - 14, 0.46, COL_GRAY, 1)
    return panel
