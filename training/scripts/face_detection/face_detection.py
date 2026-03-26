from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe import Image as MpImage
except Exception:
    mp_python = None
    mp_vision = None
    MpImage = None
    logger.warning("mediapipe tasks import failed — face landmarks unavailable")

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")


class FaceDetector:
    """MediaPipe FaceLandmarker wrapper.

    Returns landmarks as a list of numpy arrays [x, y, z] (normalized)
    for consistent, framework-agnostic downstream processing.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.detector = None

        if mp_python is None or mp_vision is None or MpImage is None:
            logger.error("mediapipe tasks not available — face detection disabled")
            return

        if model_path is None:
            model_dir = os.environ.get("MODEL_BASE_DIR", _DEFAULT_MODEL_DIR)
            model_path = os.path.join(model_dir, "face_landmarker.task")

        if not os.path.exists(model_path):
            logger.error("face_landmarker.task not found at %s", model_path)
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self.detector = mp_vision.FaceLandmarker.create_from_options(options)
            logger.info("FaceDetector initialised (model=%s)", model_path)
        except Exception as exc:
            logger.error("FaceLandmarker initialisation failed: %s", exc, exc_info=True)
            self.detector = None

    def get_landmarks(
        self, frame: np.ndarray
    ) -> Tuple[Optional[List[np.ndarray]], Optional[int], Optional[int]]:
        """Run landmark detection on a BGR frame.

        Returns (landmarks_list, img_w, img_h) or (None, None, None).
        Each landmark is a numpy array [x, y, z] with normalized coordinates.
        """
        if self.detector is None:
            return None, None, None

        if frame is None or frame.size == 0:
            return None, None, None

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MpImage(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None, None, None

        face = result.face_landmarks[0]
        landmarks = [
            np.array([lm.x, lm.y, lm.z], dtype=np.float32) for lm in face
        ]
        return landmarks, img_w, img_h
