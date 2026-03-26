"""
Parse error payloads from HTTP clients (FastAPI returns JSON ``detail``).
"""

from __future__ import annotations

import httpx

from utils.logger import get_logger

logger = get_logger(__name__)


def httpx_response_detail(response: httpx.Response) -> str:
    """Return FastAPI ``detail`` when present; otherwise raw response text."""
    body = response.text
    try:
        data = response.json()
        return str(data.get("detail", body))
    except Exception as exc:
        logger.debug("Non-JSON error body (status=%s): %s", response.status_code, exc)
        return body
