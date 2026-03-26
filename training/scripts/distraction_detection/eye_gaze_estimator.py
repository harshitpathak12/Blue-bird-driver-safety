from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

RIGHT_EYE_IDX = {"LEFT": 33, "RIGHT": 133, "TOP": 159, "BOTTOM": 145, "IRIS": 468}
LEFT_EYE_IDX = {"LEFT": 362, "RIGHT": 263, "TOP": 386, "BOTTOM": 374, "IRIS": 473}


class EyeGazeEstimator:
    """Continuous iris position ratios (0-1) for horizontal and vertical gaze."""

    def __init__(self, smoothing_alpha: float = 0.6, buffer_size: int = 5) -> None:
        self.alpha = smoothing_alpha
        self.buffer_size = buffer_size
        self._h_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._v_buffer: Deque[float] = deque(maxlen=buffer_size)
        self._last_h: Optional[float] = None
        self._last_v: Optional[float] = None
        self._max_step = 0.12

    @staticmethod
    def _xy(landmarks: List[np.ndarray], idx: int, w: int, h: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm[0] * w, lm[1] * h], dtype=np.float32)

    def _eye_ratios(
        self, landmarks: List[np.ndarray], w: int, h: int, idx_map: dict,
    ) -> Optional[Tuple[float, float]]:
        left = self._xy(landmarks, idx_map["LEFT"], w, h)
        right = self._xy(landmarks, idx_map["RIGHT"], w, h)
        top = self._xy(landmarks, idx_map["TOP"], w, h)
        bottom = self._xy(landmarks, idx_map["BOTTOM"], w, h)
        iris = self._xy(landmarks, idx_map["IRIS"], w, h)

        eye_width = np.linalg.norm(right - left)
        eye_height = np.linalg.norm(bottom - top)
        if eye_width <= 1e-6 or eye_height <= 1e-6:
            return None

        h_ratio = float((iris[0] - left[0]) / eye_width)
        v_ratio = float((iris[1] - top[1]) / eye_height)
        return h_ratio, v_ratio

    def process(
        self, landmarks: List[np.ndarray] | None,
        img_w: int | None, img_h: int | None,
    ) -> Optional[Tuple[float, float]]:
        if landmarks is None or img_w is None or img_h is None:
            return None

        right = self._eye_ratios(landmarks, img_w, img_h, RIGHT_EYE_IDX)
        left = self._eye_ratios(landmarks, img_w, img_h, LEFT_EYE_IDX)
        if right is None or left is None:
            return None

        avg_h = float(np.clip((right[0] + left[0]) / 2.0, 0.0, 1.0))
        avg_v = float(np.clip((right[1] + left[1]) / 2.0, 0.0, 1.0))

        if self._last_h is not None:
            if abs(avg_h - self._last_h) > self._max_step:
                avg_h = self._last_h + float(np.clip(avg_h - self._last_h, -self._max_step, self._max_step))
            if abs(avg_v - self._last_v) > self._max_step:
                avg_v = self._last_v + float(np.clip(avg_v - self._last_v, -self._max_step, self._max_step))

        if self._last_h is None:
            smoothed_h, smoothed_v = avg_h, avg_v
        else:
            smoothed_h = self.alpha * avg_h + (1.0 - self.alpha) * self._last_h
            smoothed_v = self.alpha * avg_v + (1.0 - self.alpha) * self._last_v

        self._last_h, self._last_v = smoothed_h, smoothed_v
        self._h_buffer.append(smoothed_h)
        self._v_buffer.append(smoothed_v)

        return float(sum(self._h_buffer) / len(self._h_buffer)), float(sum(self._v_buffer) / len(self._v_buffer))
