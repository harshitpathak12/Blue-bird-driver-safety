"""Sustained Deviation Classifier — rule-based distraction detector with hysteresis."""

from __future__ import annotations

from enum import Enum
from typing import Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

HEAD_OFF_DEG_ON = 28.0
HEAD_OFF_DEG_OFF = 22.0
GAZE_OFF_ON = 0.18
GAZE_OFF_OFF = 0.14
SUSTAINED_OFF_FRAMES = 30


class GeometricTemporalState(Enum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    WARMING_UP = "warming_up"


class GeometricTemporalClassifier:
    """Distracted only when SUSTAINED_OFF_FRAMES consecutive frames are off."""

    def __init__(self) -> None:
        self._consecutive_off = 0
        self._off_latched = False

    def push(
        self,
        head_dev_yaw: float,
        head_dev_pitch: float,
        gaze_dev_h: float,
        gaze_dev_v: float,
        alignment: float,
    ) -> None:
        head_mag = max(abs(head_dev_yaw), abs(head_dev_pitch))
        gaze_mag = max(abs(gaze_dev_h), abs(gaze_dev_v))

        if not self._off_latched:
            self._off_latched = bool(head_mag >= HEAD_OFF_DEG_ON or gaze_mag >= GAZE_OFF_ON)
        else:
            head_back = head_mag <= HEAD_OFF_DEG_OFF
            gaze_back = gaze_mag <= GAZE_OFF_OFF
            self._off_latched = not (head_back and gaze_back)

        if self._off_latched:
            self._consecutive_off += 1
        else:
            self._consecutive_off = max(0, self._consecutive_off - 3)

    def predict(self) -> Tuple[GeometricTemporalState, float]:
        if self._consecutive_off >= SUSTAINED_OFF_FRAMES:
            ratio = min(1.0, self._consecutive_off / (SUSTAINED_OFF_FRAMES * 2))
            return GeometricTemporalState.DISTRACTED, float(ratio)
        return GeometricTemporalState.ATTENTIVE, 1.0 - (self._consecutive_off / SUSTAINED_OFF_FRAMES)

    def clear_buffer(self) -> None:
        self._consecutive_off = 0
        self._off_latched = False

    @property
    def enabled(self) -> bool:
        return True
