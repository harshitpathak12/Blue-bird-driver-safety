"""Session repository — manages driving sessions."""

from datetime import datetime, timezone

from bson import ObjectId

from database.mongodb_client import sessions_collection
from utils.logger import get_logger

logger = get_logger(__name__)


class SessionRepository:
    """Class-based session data access."""

    @staticmethod
    def create(driver_id: str) -> str:
        session = {
            "driver_id": driver_id,
            "start_time": datetime.now(timezone.utc),
            "end_time": None,
            "status": "active",
        }
        try:
            result = sessions_collection.insert_one(session)
            logger.info("Session created: id=%s driver=%s", result.inserted_id, driver_id)
            return str(result.inserted_id)
        except Exception as e:
            logger.error("SessionRepository.create failed: %s", e, exc_info=True)
            raise

    @staticmethod
    def get(session_id: str) -> dict | None:
        try:
            return sessions_collection.find_one({"_id": ObjectId(session_id)})
        except Exception as e:
            logger.warning("SessionRepository.get failed for %s: %s", session_id, e)
            return None

    @staticmethod
    def end(session_id: str) -> None:
        try:
            sessions_collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"end_time": datetime.now(timezone.utc), "status": "ended"}},
            )
            logger.info("Session ended: %s", session_id)
        except Exception as e:
            logger.error("SessionRepository.end failed for %s: %s", session_id, e)


def create_session(driver_id):
    return SessionRepository.create(driver_id)


def get_session(session_id):
    return SessionRepository.get(session_id)


def end_session(session_id):
    SessionRepository.end(session_id)
