"""
Visualization utilities for driving license detection.

- Draw bounding boxes on images (in-place for speed)
- Draw validation status banner
"""

from typing import Sequence

import cv2
import numpy as np


def draw_bbox(
    image: np.ndarray,
    bbox: Sequence[float],
    label: str = "driving_license",
    confidence: float | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box on image in-place. Returns same array."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    text = label
    if confidence is not None:
        text += f" {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
    cv2.putText(
        image, text, (x1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    return image


def draw_validation_status(
    image: np.ndarray,
    label: str,
    reason: str = "",
) -> np.ndarray:
    """Draw a prominent status banner at top of frame. Modifies in-place."""
    h, w = image.shape[:2]
    text = label.upper()
    if reason:
        text += f" - {reason[:40]}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    pad = 10
    y2 = th + 2 * pad
    cv2.rectangle(image, (0, 0), (w, y2), (0, 0, 0), -1)
    color = (
        (0, 255, 0) if label.lower() == "valid"
        else (0, 0, 255) if label.lower() == "invalid"
        else (0, 255, 255) if label.lower() == "processing"
        else (0, 165, 255) if label.lower() == "no_text"
        else (200, 200, 200)
    )
    cv2.putText(image, text, (pad, th + pad), font, 0.8, color, 2)
    return image


