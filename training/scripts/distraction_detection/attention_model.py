from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class AttentionState(Enum):
    CALIBRATING = "calibrating"
    ATTENTIVE = "attentive"
    DELIBERATE_LOOK = "deliberate_look"
    DISTRACTED = "distracted"


@dataclass
class AttentionResult:
    state: AttentionState
    head_deviation_yaw: float = 0.0
    head_deviation_pitch: float = 0.0
    gaze_deviation_h: float = 0.0
    gaze_deviation_v: float = 0.0
    alignment_score: float = 1.0
    distraction_duration_sec: float = 0.0
    calibrated: bool = False
    calibration_elapsed_sec: float = 0.0
    calibration_remaining_sec: float = 0.0


HEAD_DEVIATION_THRESHOLD = 8.0
GAZE_DEVIATION_THRESHOLD = 0.06
ALIGNMENT_THRESHOLD = 0.5
DISTRACTION_CONFIRM_SEC = 1.0
DELIBERATE_LOOK_MAX_SEC = 5.0
CALIBRATION_DURATION_SEC = 5.0
MIN_CALIBRATION_SAMPLES = 50


class AttentionModel:
    """Baseline-calibrated attention model robust to camera mounting angle."""

    def __init__(self) -> None:
        self._calibrating: bool = True
        self._calibration_start: Optional[float] = None
        self._head_samples: List[Tuple[float, float]] = []
        self._gaze_samples: List[Tuple[float, float]] = []

        self.baseline_yaw: float = 0.0
        self.baseline_pitch: float = 0.0
        self.baseline_gaze_h: float = 0.5
        self.baseline_gaze_v: float = 0.5

        self._distraction_start: Optional[float] = None
        self._deliberate_start: Optional[float] = None
        self._distraction_duration: float = 0.0
        self._dev_buffer: Deque[Tuple[float, float, float, float]] = deque(maxlen=5)

    def recalibrate(self) -> None:
        self._calibrating = True
        self._calibration_start = None
        self._head_samples.clear()
        self._gaze_samples.clear()
        self._distraction_start = None
        self._deliberate_start = None
        self._distraction_duration = 0.0
        self._dev_buffer.clear()

    @property
    def calibrated(self) -> bool:
        return not self._calibrating

    def _median(self, values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        if n % 2 == 1:
            return float(sorted_vals[mid])
        return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)

    def _finish_calibration(self) -> None:
        if not self._head_samples or not self._gaze_samples:
            return
        self.baseline_yaw = self._median([y for y, _ in self._head_samples])
        self.baseline_pitch = self._median([p for _, p in self._head_samples])
        self.baseline_gaze_h = self._median([h for h, _ in self._gaze_samples])
        self.baseline_gaze_v = self._median([v for _, v in self._gaze_samples])
        self._calibrating = False
        logger.info(
            "Calibration complete: baseline yaw=%.1f pitch=%.1f gaze_h=%.3f gaze_v=%.3f (%d samples)",
            self.baseline_yaw, self.baseline_pitch, self.baseline_gaze_h, self.baseline_gaze_v,
            len(self._head_samples),
        )

    def nudge_baseline(
        self,
        current_yaw: float,
        current_pitch: float,
        current_gaze_h: float,
        current_gaze_v: float,
        alpha: float = 0.008,
    ) -> None:
        if self._calibrating:
            return
        inv = 1.0 - alpha
        self.baseline_yaw = inv * self.baseline_yaw + alpha * current_yaw
        self.baseline_pitch = inv * self.baseline_pitch + alpha * current_pitch
        self.baseline_gaze_h = inv * self.baseline_gaze_h + alpha * current_gaze_h
        self.baseline_gaze_v = inv * self.baseline_gaze_v + alpha * current_gaze_v

    def _alignment_score(
        self, head_dev_yaw: float, head_dev_pitch: float,
        gaze_dev_h: float, gaze_dev_v: float,
    ) -> float:
        head_vec = np.array([head_dev_yaw, -head_dev_pitch], dtype=float)
        gaze_vec = np.array([gaze_dev_h, -gaze_dev_v], dtype=float)
        head_mag = float(np.linalg.norm(head_vec))
        gaze_mag = float(np.linalg.norm(gaze_vec))
        if head_mag < 1e-6 and gaze_mag < 1e-6:
            return 1.0
        if head_mag < 1e-6 or gaze_mag < 1e-6:
            mag = max(head_mag, gaze_mag)
            return max(0.0, 1.0 - mag / max(HEAD_DEVIATION_THRESHOLD, GAZE_DEVIATION_THRESHOLD))
        cos_sim = float(np.dot(head_vec, gaze_vec) / (head_mag * gaze_mag))
        return max(0.0, min(1.0, 0.5 * (cos_sim + 1.0)))

    def process(
        self, head_pitch: float, head_yaw: float,
        gaze_avg_h: float, gaze_avg_v: float,
    ) -> AttentionResult:
        now = time.time()

        if self._calibrating:
            if self._calibration_start is None:
                self._calibration_start = now
            self._head_samples.append((head_yaw, head_pitch))
            self._gaze_samples.append((gaze_avg_h, gaze_avg_v))
            duration = now - self._calibration_start
            remaining = max(0.0, CALIBRATION_DURATION_SEC - duration)
            if duration >= CALIBRATION_DURATION_SEC and len(self._head_samples) >= MIN_CALIBRATION_SAMPLES:
                self._finish_calibration()
            return AttentionResult(
                state=AttentionState.CALIBRATING,
                calibrated=self.calibrated,
                calibration_elapsed_sec=duration,
                calibration_remaining_sec=remaining,
            )

        head_dev_yaw = head_yaw - self.baseline_yaw
        head_dev_pitch = head_pitch - self.baseline_pitch
        gaze_dev_h = gaze_avg_h - self.baseline_gaze_h
        gaze_dev_v = gaze_avg_v - self.baseline_gaze_v

        self._dev_buffer.append((head_dev_yaw, head_dev_pitch, gaze_dev_h, gaze_dev_v))
        arr = np.array(list(self._dev_buffer), dtype=float)
        head_dev_yaw_s, head_dev_pitch_s, gaze_dev_h_s, gaze_dev_v_s = arr.mean(axis=0)

        head_moved = abs(head_dev_yaw_s) > HEAD_DEVIATION_THRESHOLD or abs(head_dev_pitch_s) > HEAD_DEVIATION_THRESHOLD
        gaze_moved = abs(gaze_dev_h_s) > GAZE_DEVIATION_THRESHOLD or abs(gaze_dev_v_s) > GAZE_DEVIATION_THRESHOLD
        alignment = self._alignment_score(head_dev_yaw_s, head_dev_pitch_s, gaze_dev_h_s, gaze_dev_v_s)

        state = AttentionState.ATTENTIVE
        if not head_moved and not gaze_moved:
            self._distraction_start = None
            self._deliberate_start = None
            self._distraction_duration = 0.0
        else:
            if self._distraction_start is None:
                self._distraction_start = now
            self._deliberate_start = None
            self._distraction_duration = now - self._distraction_start
            if self._distraction_duration >= DISTRACTION_CONFIRM_SEC:
                state = AttentionState.DISTRACTED

        return AttentionResult(
            state=state,
            head_deviation_yaw=float(head_dev_yaw_s),
            head_deviation_pitch=float(head_dev_pitch_s),
            gaze_deviation_h=float(gaze_dev_h_s),
            gaze_deviation_v=float(gaze_dev_v_s),
            alignment_score=float(alignment),
            distraction_duration_sec=float(self._distraction_duration),
            calibrated=self.calibrated,
        )
