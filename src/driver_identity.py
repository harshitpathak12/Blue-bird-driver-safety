"""Driver identity matching service."""

import os
from typing import Optional, Tuple

import numpy as np

from database import driver_repository
from utils.logger import get_logger

logger = get_logger(__name__)


class DriverIdentityService:
    """
    Class-based driver identity matcher.

    1:N login uses cosine similarity. Wrong-name logins often happen when:
    - the threshold is too low (many people pass), or
    - only the argmax is used (no check that the runner-up is clearly worse).

    We address this by:
    - fusing 2D (ArcFace) + 3D embeddings when both exist (2D is more discriminative),
    - requiring a minimum gap between best and second-best match (ambiguous → reject),
    - stricter default threshold (configurable via env).
    """

    # Cosine similarity; 3D geometry-only embeddings can overlap across people — use higher default.
    RECOGNITION_THRESHOLD = float(os.environ.get("DRIVER_RECOGNITION_THRESHOLD", "0.52"))
    # If top-two scores are within this margin, treat as ambiguous (no match).
    MIN_TOP2_MARGIN = float(os.environ.get("DRIVER_RECOGNITION_MARGIN", "0.065"))
    # When both 2D and 3D exist for query and DB row, blend (weight on ArcFace).
    WEIGHT_2D = float(os.environ.get("DRIVER_MATCH_WEIGHT_2D", "0.65"))

    @staticmethod
    def _cosine_score(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.shape != b.shape or a.size == 0:
            return -1.0
        an = np.linalg.norm(a) + 1e-10
        bn = np.linalg.norm(b) + 1e-10
        return float(np.dot(a, b) / (an * bn))

    @classmethod
    def _score_against_driver(
        cls,
        driver: dict,
        emb_2d: Optional[np.ndarray],
        emb_3d: Optional[np.ndarray],
    ) -> float:
        """Single similarity score in [ -1, 1 ], or -1 if no comparable modality."""
        stored_3d = driver.get("face_embedding_3d")
        stored_2d = driver.get("face_embedding")

        s3: Optional[float] = None
        s2: Optional[float] = None

        if stored_3d is not None and emb_3d is not None:
            db3 = np.asarray(stored_3d, dtype=np.float32)
            q3 = np.asarray(emb_3d, dtype=np.float32)
            s3 = cls._cosine_score(q3, db3)

        if stored_2d is not None and emb_2d is not None:
            db2 = np.asarray(stored_2d, dtype=np.float32)
            q2 = np.asarray(emb_2d, dtype=np.float32)
            s2 = cls._cosine_score(q2, db2)

        if s3 is not None and s2 is not None:
            w2 = cls.WEIGHT_2D
            w3 = 1.0 - w2
            return w3 * s3 + w2 * s2
        if s3 is not None:
            return s3
        if s2 is not None:
            return s2
        return -1.0

    @classmethod
    def match(
        cls,
        embedding: Optional[np.ndarray] = None,
        embedding_2d: Optional[np.ndarray] = None,
        embedding_3d: Optional[np.ndarray] = None,
        driver_id: Optional[str] = None,
    ) -> Tuple[Optional[dict], float]:
        emb_2d = embedding_2d if embedding_2d is not None else embedding
        emb_2d = np.asarray(emb_2d, dtype=np.float32) if emb_2d is not None else None
        emb_3d = np.asarray(embedding_3d, dtype=np.float32) if embedding_3d is not None else None

        # 1:1 verification (optional driver_id)
        if driver_id:
            driver = driver_repository.get_driver_by_id(driver_id)
            if not driver:
                logger.debug("No driver found for id=%s", driver_id)
                return None, -1.0
            score = cls._score_against_driver(driver, emb_2d, emb_3d)
            if score < 0:
                return None, -1.0
            ok = score >= cls.RECOGNITION_THRESHOLD
            return (driver if ok else None), score

        # 1:N identification — collect all scores, then apply threshold + top-2 margin
        scored: list[tuple[dict, float]] = []
        for driver in driver_repository.get_all_drivers():
            score = cls._score_against_driver(driver, emb_2d, emb_3d)
            if score >= 0:
                scored.append((driver, score))

        if not scored:
            logger.debug("No driver with comparable embeddings")
            return None, -1.0

        scored.sort(key=lambda x: -x[1])
        best_driver, best_score = scored[0]

        if best_score < cls.RECOGNITION_THRESHOLD:
            logger.debug(
                "No match above threshold (best=%.3f, threshold=%.3f)",
                best_score,
                cls.RECOGNITION_THRESHOLD,
            )
            return None, best_score

        if len(scored) >= 2:
            second_score = scored[1][1]
            gap = best_score - second_score
            if gap < cls.MIN_TOP2_MARGIN:
                logger.info(
                    "Ambiguous face match — rejecting (best=%.3f second=%.3f gap=%.3f need>=%.3f)",
                    best_score,
                    second_score,
                    gap,
                    cls.MIN_TOP2_MARGIN,
                )
                return None, best_score

        logger.info(
            "Driver matched: %s (score=%.3f)",
            best_driver.get("driver_id"),
            best_score,
        )
        return best_driver, best_score


def match_embedding_to_driver(
    embedding: Optional[np.ndarray] = None,
    embedding_2d: Optional[np.ndarray] = None,
    embedding_3d: Optional[np.ndarray] = None,
    driver_id: Optional[str] = None,
) -> Tuple[Optional[dict], float]:
    """Compatibility function wrapper."""
    return DriverIdentityService.match(
        embedding=embedding,
        embedding_2d=embedding_2d,
        embedding_3d=embedding_3d,
        driver_id=driver_id,
    )
