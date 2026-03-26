"""Centralized logging for the Driver Safety System.

Usage in any module::

    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Server started on port %d", port)
"""

from __future__ import annotations

import logging
import logging.config
import os
from typing import Any, Dict

import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOG_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "configs", "logging.yml")
_INITIALIZED = False


def _resolve_log_paths(config: Dict[str, Any]) -> None:
    """Convert relative ``filename`` entries in handlers to absolute paths."""
    for handler_cfg in config.get("handlers", {}).values():
        if "filename" in handler_cfg:
            handler_cfg["filename"] = os.path.join(
                _PROJECT_ROOT, handler_cfg["filename"]
            )


def setup_logging() -> None:
    """Load YAML logging config and create the log directory.

    Safe to call multiple times; only the first invocation takes effect.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    log_dir = os.path.join(_PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if os.path.exists(_LOG_CONFIG_PATH):
        with open(_LOG_CONFIG_PATH, "r") as fh:
            config = yaml.safe_load(fh)
        _resolve_log_paths(config)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising the config on first call."""
    setup_logging()
    return logging.getLogger(name)
