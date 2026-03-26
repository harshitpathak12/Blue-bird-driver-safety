"""Alert repository — stores and queries alert records."""

from datetime import datetime, timezone

from database.mongodb_client import alerts_collection
from utils.logger import get_logger

logger = get_logger(__name__)


class AlertRepository:
    """Class-based alert data access."""

    @staticmethod
    def insert(
        driver_id: str,
        alert_type: str,
        confidence_score: float,
        session_id: str | None = None,
        gps_latitude: float | None = None,
        gps_longitude: float | None = None,
    ) -> dict:
        alert = {
            "driver_id": driver_id,
            "session_id": session_id,
            "alert_type": alert_type,
            "confidence_score": confidence_score,
            "timestamp": datetime.now(timezone.utc),
        }
        if gps_latitude is not None and gps_longitude is not None:
            alert["gps"] = {"latitude": gps_latitude, "longitude": gps_longitude}
        try:
            result = alerts_collection.insert_one(alert)
            alert["_id"] = result.inserted_id
            logger.debug("Alert inserted: id=%s driver=%s type=%s", result.inserted_id, driver_id, alert_type)
            return alert
        except Exception as e:
            logger.error("AlertRepository.insert failed: %s", e, exc_info=True)
            raise

    @staticmethod
    def get_alerts(driver_id: str | None = None, session_id: str | None = None, limit: int = 100) -> list:
        query = {}
        if driver_id:
            query["driver_id"] = driver_id
        if session_id:
            query["session_id"] = session_id
        cursor = alerts_collection.find(query).sort("timestamp", -1).limit(limit)
        return list(cursor)


def insert_alert(driver_id, alert_type, confidence_score, session_id=None, gps_latitude=None, gps_longitude=None):
    return AlertRepository.insert(driver_id, alert_type, confidence_score, session_id, gps_latitude, gps_longitude)


def get_alerts(driver_id=None, session_id=None, limit=100):
    return AlertRepository.get_alerts(driver_id, session_id, limit)
