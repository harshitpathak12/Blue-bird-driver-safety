"""
Image utilities for driving license detection.

- Crop by bounding box with clipping and optional padding
"""

from typing import Sequence

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def crop_bbox(
    image: np.ndarray,
    bbox: Sequence[float],
    padding: int = 0,
    min_size: int = 32,
) -> np.ndarray | None:
    """
    Crop image to bounding box with clipping and optional padding.

    Args:
        image: HWC numpy image (BGR).
        bbox: [x1, y1, x2, y2] in pixel coordinates.
        padding: Pixels to add around the box (before clipping).
        min_size: Minimum width/height; if crop is smaller, pad with zeros.

    Returns:
        Cropped image (BGR), or None if invalid/empty crop.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    if padding:
        x1 = x1 - padding
        y1 = y1 - padding
        x2 = x2 + padding
        y2 = y2 + padding
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        logger.debug("crop_bbox: empty region after clip (x1,y1,x2,y2)=(%s,%s,%s,%s)", x1, y1, x2, y2)
        return None

    crop = image[y1:y2, x1:x2].copy()

    cw, ch = crop.shape[1], crop.shape[0]
    if cw < min_size or ch < min_size:
        # Pad to at least min_size
        nw = max(min_size, cw)
        nh = max(min_size, ch)
        padded = np.zeros((nh, nw, crop.shape[2]), dtype=crop.dtype)
        padded[:] = 255  # white background for document-like crop
        padded[:ch, :cw] = crop
        crop = padded

    return crop
