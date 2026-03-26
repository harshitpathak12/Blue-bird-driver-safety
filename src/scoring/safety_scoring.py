"""Daily safety scoring service."""

from utils.logger import get_logger

logger = get_logger(__name__)


def risk_level_from_score(score: float) -> str:
    if score >= 90:
        return "Safe"
    if score >= 70:
        return "Moderate Risk"
    return "High Risk"


class SafetyScoring:
    FATIGUE_PENALTY = 5
    DISTRACTION_PENALTY = 3
    SLEEP_PENALTY = 10

    @classmethod
    def compute(cls, fatigue_count: int, distraction_count: int, sleep_count: int) -> float:
        score = 100.0
        score -= fatigue_count * cls.FATIGUE_PENALTY
        score -= distraction_count * cls.DISTRACTION_PENALTY
        score -= sleep_count * cls.SLEEP_PENALTY
        result = max(0.0, min(100.0, score))
        logger.debug("Safety score: %.1f (fatigue=%d distraction=%d sleep=%d)", result, fatigue_count, distraction_count, sleep_count)
        return result

