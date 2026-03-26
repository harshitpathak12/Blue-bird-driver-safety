"""
Heavy model preload at FastAPI startup (YOLO + Qwen for /api/login/verify-dl).

The realtime monitoring pipeline (``RealtimeFramePipeline``) already loads when
``realtime_monitoring`` is imported. DL verification weights are otherwise lazy
on first ``POST /verify-dl`` — this module warms them when the server starts.

Set ``SKIP_SERVER_MODEL_PRELOAD=1`` to skip (faster dev startup if you never hit verify-dl).
"""

from __future__ import annotations

import os

from utils.logger import get_logger

logger = get_logger(__name__)


def preload_heavy_models() -> None:
    if os.environ.get("SKIP_SERVER_MODEL_PRELOAD", "").strip().lower() in ("1", "true", "yes", "on"):
        logger.info("SKIP_SERVER_MODEL_PRELOAD set — skipping DL YOLO/Qwen preload")
        return

    try:
        from training.scripts.driving_license_model.dl_validator import get_yolo

        get_yolo()
        logger.info("DL YOLO weights loaded at server startup")
    except Exception as e:
        logger.warning("DL YOLO preload failed (verify-dl may load later): %s", e, exc_info=True)

    try:
        from training.scripts.driving_license_model.qwen_ocr import ensure_qwen_loaded

        ensure_qwen_loaded()
        logger.info("Qwen VL model loaded at server startup")
    except Exception as e:
        logger.warning("Qwen preload failed (verify-dl may load later): %s", e, exc_info=True)
