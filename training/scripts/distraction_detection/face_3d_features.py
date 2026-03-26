"""3D face feature extraction for camera-mount-invariant driver attention."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

NOSE_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263

KEY_3D_INDICES = [
    33, 263, 1, 61, 291, 199,
    362, 133, 386, 374, 159, 145,
    10, 152, 176,
    13, 14, 78, 308,
    70, 300, 109, 338, 0,
]


def extract_face_3d_features(landmarks: List[np.ndarray]) -> Optional[np.ndarray]:
    """Build face-centric, scale-normalized 3D feature vector (72-dim)."""
    if landmarks is None or len(landmarks) < max(KEY_3D_INDICES) + 1:
        return None

    try:
        nose = np.array([landmarks[NOSE_TIP_IDX][0], landmarks[NOSE_TIP_IDX][1], landmarks[NOSE_TIP_IDX][2]], dtype=np.float32)
        left_eye = np.array([landmarks[LEFT_EYE_OUTER_IDX][0], landmarks[LEFT_EYE_OUTER_IDX][1], landmarks[LEFT_EYE_OUTER_IDX][2]], dtype=np.float32)
        right_eye = np.array([landmarks[RIGHT_EYE_OUTER_IDX][0], landmarks[RIGHT_EYE_OUTER_IDX][1], landmarks[RIGHT_EYE_OUTER_IDX][2]], dtype=np.float32)
    except IndexError:
        return None

    scale = float(np.linalg.norm(left_eye - right_eye))
    if scale < 1e-6:
        scale = 1.0

    points_3d = []
    for idx in KEY_3D_INDICES:
        pt = np.array([landmarks[idx][0], landmarks[idx][1], landmarks[idx][2]], dtype=np.float32)
        pt = (pt - nose) / scale
        points_3d.append(pt)

    return np.concatenate(points_3d).astype(np.float32)
