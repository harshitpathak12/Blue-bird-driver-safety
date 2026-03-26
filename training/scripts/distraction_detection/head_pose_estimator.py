from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

LANDMARK_IDS = [33, 263, 1, 61, 291, 199]


class HeadPoseEstimator:
    """Estimate head pose (pitch, yaw, roll) via cv2.solvePnP."""

    def __init__(self) -> None:
        self.last_pitch: float = 0.0
        self.last_yaw: float = 0.0
        self.last_roll: float = 0.0

    def _build_model_points(self) -> np.ndarray:
        return np.array(
            [
                [-30.0, 0.0, 30.0],
                [30.0, 0.0, 30.0],
                [0.0, 0.0, 60.0],
                [-25.0, -40.0, 30.0],
                [25.0, -40.0, 30.0],
                [0.0, -75.0, 0.0],
            ],
            dtype=np.float32,
        )

    def process(
        self, landmarks: List[np.ndarray] | None,
        img_w: int | None, img_h: int | None,
    ) -> Optional[Tuple[float, float, float]]:
        if landmarks is None or img_w is None or img_h is None:
            return None

        try:
            image_points = []
            for idx in LANDMARK_IDS:
                lm = landmarks[idx]
                image_points.append([lm[0] * img_w, lm[1] * img_h])

            image_points = np.asarray(image_points, dtype=np.float32)
            model_points = self._build_model_points()

            focal_length = float(img_w)
            center = (img_w / 2.0, img_h / 2.0)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype=np.float32,
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            success, rvec, _tvec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return None

            rot_mat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
            pitch, yaw, roll = [float(a) for a in angles]

            pitch = float(np.clip(pitch, -90.0, 90.0))
            yaw = float(np.clip(yaw, -90.0, 90.0))
            roll = float(np.clip(roll, -90.0, 90.0))

            self.last_pitch, self.last_yaw, self.last_roll = pitch, yaw, roll
            return pitch, yaw, roll
        except Exception:
            return self.last_pitch, self.last_yaw, self.last_roll
