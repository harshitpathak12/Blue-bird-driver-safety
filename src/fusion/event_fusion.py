"""
Multi-Model Fusion Engine.
Combines fatigue, distraction, blink/PERCLOS into unified driver state and alerts.
"""

from dataclasses import dataclass
from typing import Literal

from utils.logger import get_logger

logger = get_logger(__name__)

AlertType = Literal["fatigue", "distraction", "sleep"]


@dataclass
class ModelOutputs:
    perclos: float = 0.0
    blink_duration_sec: float = 0.0
    blink_rate_low: bool = False
    fatigue_score: float = 0.0
    head_turned_away_sec: float = 0.0
    distraction_score: float = 0.0
    eye_closure_duration_sec: float = 0.0


@dataclass
class FusionResult:
    driver_state: str
    alert_type: AlertType | None = None
    confidence_score: float = 0.0
    message: str = ""


class FusionEngine:
    def __init__(
        self,
        perclos_sleep_threshold: float = 0.4,
        eye_closure_sleep_sec: float = 2.0,
        head_turned_distraction_sec: float = 3.0,
        fatigue_threshold: float = 0.6,
    ):
        self.perclos_sleep_threshold = perclos_sleep_threshold
        self.eye_closure_sleep_sec = eye_closure_sleep_sec
        self.head_turned_distraction_sec = head_turned_distraction_sec
        self.fatigue_threshold = fatigue_threshold

    def fuse(self, outputs: ModelOutputs) -> FusionResult:
        if (
            outputs.perclos >= self.perclos_sleep_threshold
            and outputs.eye_closure_duration_sec >= self.eye_closure_sleep_sec
        ):
            logger.debug("Fusion → SLEEP (perclos=%.2f, eye_closure=%.1fs)", outputs.perclos, outputs.eye_closure_duration_sec)
            return FusionResult(
                driver_state="sleep",
                alert_type="sleep",
                confidence_score=min(1.0, outputs.perclos + 0.2),
                message="Prolonged eye closure detected",
            )
        if outputs.head_turned_away_sec >= self.head_turned_distraction_sec:
            logger.debug("Fusion → DISTRACTION (head_away=%.1fs)", outputs.head_turned_away_sec)
            return FusionResult(
                driver_state="distraction",
                alert_type="distraction",
                confidence_score=min(1.0, outputs.distraction_score or 0.8),
                message="Head turned away from road",
            )
        if outputs.fatigue_score >= self.fatigue_threshold or outputs.blink_rate_low:
            logger.debug("Fusion → FATIGUE (score=%.2f, blink_low=%s)", outputs.fatigue_score, outputs.blink_rate_low)
            return FusionResult(
                driver_state="fatigue",
                alert_type="fatigue",
                confidence_score=outputs.fatigue_score or 0.7,
                message="Fatigue indicators detected",
            )
        return FusionResult(driver_state="normal", message="")

