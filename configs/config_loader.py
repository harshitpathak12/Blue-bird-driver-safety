"""Configuration loader — reads YAML config once and caches."""

import os
from typing import Any, Dict

import yaml

from utils.logger import get_logger

_logger = get_logger(__name__)


class ConfigLoader:
    """Singleton config loader for the application."""

    _config: Dict[str, Any] | None = None
    _config_dir = os.path.abspath(os.path.dirname(__file__))

    @classmethod
    def load(cls, path: str | None = None) -> Dict[str, Any]:
        if cls._config is not None:
            return cls._config
        if path is None:
            path = os.path.join(cls._config_dir, "config.yaml")
        _logger.debug("Loading config from %s", path)
        with open(path, "r") as f:
            cls._config = yaml.safe_load(f)
        _logger.info("Config loaded (%d top-level keys)", len(cls._config))
        return cls._config

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        cfg = cls.load()
        return cfg.get(key, default)


config = ConfigLoader.load()
