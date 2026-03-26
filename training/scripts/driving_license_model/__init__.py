"""
Driving licence detection (YOLO + Qwen OCR + Indian DL rules).

Do not import ``dl_validator`` here: it pulls in ``torch``/ultralytics. Import it only where needed
(``from training.scripts.driving_license_model.dl_validator import validate_license_frame``).
"""

from .dl_matching import registration_matches_dl

__all__ = ["registration_matches_dl"]
