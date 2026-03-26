"""Daily safety scores repository."""

from datetime import date, datetime, timezone

from database.mongodb_client import daily_scores_collection
from utils.logger import get_logger

logger = get_logger(__name__)


class DailyScoresRepository:
    """Class-based daily scores data access."""

    @staticmethod
    def upsert(
        driver_id: str,
        score_date: date,
        fatigue_count: int = 0,
        distraction_count: int = 0,
        sleep_count: int = 0,
        safety_score: float = 0.0,
    ) -> dict:
        doc = {
            "driver_id": driver_id,
            "date": score_date.isoformat(),
            "fatigue_count": fatigue_count,
            "distraction_count": distraction_count,
            "sleep_count": sleep_count,
            "safety_score": safety_score,
            "updated_at": datetime.now(timezone.utc),
        }
        daily_scores_collection.update_one(
            {"driver_id": driver_id, "date": score_date.isoformat()},
            {"$set": doc},
            upsert=True,
        )
        logger.debug("Daily score upserted: driver=%s date=%s score=%.1f", driver_id, score_date, safety_score)
        return doc

    @staticmethod
    def get(driver_id: str, date_from: str | None = None, date_to: str | None = None) -> list:
        query = {"driver_id": driver_id}
        if date_from and date_to:
            query["date"] = {"$gte": date_from, "$lte": date_to}
        elif date_from:
            query["date"] = {"$gte": date_from}
        elif date_to:
            query["date"] = {"$lte": date_to}
        cursor = daily_scores_collection.find(query).sort("date", -1)
        return list(cursor)


def upsert_daily_score(driver_id, score_date, fatigue_count=0, distraction_count=0, sleep_count=0, safety_score=0.0):
    return DailyScoresRepository.upsert(driver_id, score_date, fatigue_count, distraction_count, sleep_count, safety_score)


def get_daily_scores(driver_id, date_from=None, date_to=None):
    return DailyScoresRepository.get(driver_id, date_from, date_to)
