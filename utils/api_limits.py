"""
Central limits and validation for HTTP/WebSocket APIs (production defaults).

Override via environment variables without code changes.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from fastapi import HTTPException

from utils.logger import get_logger

logger = get_logger(__name__)

# Login / register / verify-dl: multipart image uploads
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))

# WebSocket: inbound JPEG frame size (per message)
MAX_STREAM_FRAME_BYTES = int(os.environ.get("MAX_STREAM_FRAME_BYTES", str(4 * 1024 * 1024)))

# Local HTTP client (`data_pipeline.client`) timeouts — match server route cost
HTTP_CLIENT_TIMEOUT_LOGIN_S = float(os.environ.get("HTTP_CLIENT_TIMEOUT_LOGIN_S", "60"))
HTTP_CLIENT_TIMEOUT_REGISTER_S = float(os.environ.get("HTTP_CLIENT_TIMEOUT_REGISTER_S", "90"))
HTTP_CLIENT_TIMEOUT_VERIFY_DL_S = float(os.environ.get("HTTP_CLIENT_TIMEOUT_VERIFY_DL_S", "300"))
HTTP_CLIENT_TIMEOUT_RECALIBRATE_S = float(os.environ.get("HTTP_CLIENT_TIMEOUT_RECALIBRATE_S", "30"))

# driver_id from DB generator: uppercase A-Z + digits, length 9; allow slight slack
_DRIVER_ID_RE = re.compile(r"^[A-Z0-9]{6,32}$")


def sanitize_driver_id(raw: Optional[str]) -> Optional[str]:
    """Return normalized driver_id or None if missing/invalid."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if len(s) > 32:
        return None
    if not _DRIVER_ID_RE.match(s):
        return None
    return s


def check_upload_size(size: int, max_bytes: int = MAX_IMAGE_BYTES) -> None:
    """Reject oversized uploads with HTTP 413."""
    if size > max_bytes:
        logger.warning(
            "Upload rejected: size=%d exceeds max=%d",
            size,
            max_bytes,
        )
        raise HTTPException(
            status_code=413,
            detail=f"Payload too large ({size} bytes); maximum is {max_bytes} bytes.",
        )


def is_production_error_detail() -> bool:
    """If True, avoid leaking exception strings to API clients (set API_DEBUG=1 to disable)."""
    return os.environ.get("API_DEBUG", "").strip().lower() not in {"1", "true", "yes", "on"}


def public_error_message(internal: str, public: str = "An internal error occurred") -> str:
    """Return safe client-facing message when not in debug mode."""
    if is_production_error_detail():
        return public
    return internal
