"""Driver repository — manages driver records."""

import random
import string
from datetime import datetime, timezone

from database.mongodb_client import drivers_collection
from utils.logger import get_logger

logger = get_logger(__name__)

_ID_CHARS = string.ascii_uppercase + string.digits
_ID_LENGTH = 9


class DriverRepository:
    """Class-based driver data access."""

    @staticmethod
    def get_by_id(driver_id: str) -> dict | None:
        return drivers_collection.find_one({"driver_id": driver_id})

    @staticmethod
    def get_all():
        return drivers_collection.find()

    @staticmethod
    def _generate_unique_id() -> str:
        while True:
            candidate = "".join(random.choices(_ID_CHARS, k=_ID_LENGTH))
            if not drivers_collection.find_one({"driver_id": candidate}):
                return candidate

    @classmethod
    def create(
        cls,
        driver_id: str | None,
        name: str,
        age: int | None = None,
        face_embedding: list | None = None,
        face_embedding_3d: list | None = None,
        face_image_path: str | None = None,
    ) -> dict:
        if driver_id is None:
            driver_id = cls._generate_unique_id()
        driver = {
            "driver_id": driver_id,
            "name": name,
            "age": age,
            "face_embedding": face_embedding,
            "face_embedding_3d": face_embedding_3d,
            "face_image_path": face_image_path,
            "created_at": datetime.now(timezone.utc),
            # Driving licence (filled after /api/login/verify-dl)
            "dl_number": None,
            "dl_verified": False,
            "dl_name_on_card": None,
            "dl_validity_end": None,
            "dl_verified_at": None,
        }
        drivers_collection.insert_one(driver)
        logger.info("Driver created: id=%s name=%s", driver_id, name)
        return driver

    @staticmethod
    def update_last_seen(driver_id: str) -> None:
        drivers_collection.update_one(
            {"driver_id": driver_id},
            {"$set": {"last_seen": datetime.now(timezone.utc)}},
        )

    @staticmethod
    def update_driving_license(
        driver_id: str,
        *,
        dl_number: str,
        dl_verified: bool,
        dl_name_on_card: str | None,
        dl_validity_end: str | None,
        registration_name_match: bool,
        registration_age_match: bool,
    ) -> None:
        """Persist verified driving licence fields after successful OCR + rule checks."""
        now = datetime.now(timezone.utc)
        drivers_collection.update_one(
            {"driver_id": driver_id},
            {
                "$set": {
                    "dl_number": dl_number,
                    "dl_verified": dl_verified,
                    "dl_name_on_card": dl_name_on_card,
                    "dl_validity_end": dl_validity_end,
                    "dl_verified_at": now,
                    "registration_name_match": registration_name_match,
                    "registration_age_match": registration_age_match,
                }
            },
        )
        logger.info(
            "Driver DL updated: id=%s dl_number=%s verified=%s",
            driver_id,
            dl_number,
            dl_verified,
        )


def get_driver_by_id(driver_id):
    return DriverRepository.get_by_id(driver_id)


def get_all_drivers():
    return DriverRepository.get_all()


def create_driver(driver_id=None, name="", age=None, face_embedding=None, face_embedding_3d=None, face_image_path=None):
    return DriverRepository.create(driver_id, name, age, face_embedding, face_embedding_3d, face_image_path)


def update_last_seen(driver_id):
    DriverRepository.update_last_seen(driver_id)
