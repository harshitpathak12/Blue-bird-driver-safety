"""
3D face embedding from MediaPipe landmarks using Open3D PointCloud.

Landmarks are centered, scale-normalized (inter-eye distance), packed into an
Open3D point cloud, then flattened and L2-normalized for cosine similarity.
Same geometry as the previous pure-NumPy path to keep stored embeddings compatible.
"""

from typing import Any

import numpy as np
import open3d as o3d

from utils.logger import get_logger

logger = get_logger(__name__)


class FaceEmbeddingOpen3DBuilder:
    """Builds L2-normalized embedding vector from MediaPipe 3D landmarks via Open3D."""

    _MIN_LANDMARKS = 10
    _LEFT_EYE_CENTER_IDS = [33, 133, 159, 145, 153, 144]
    _RIGHT_EYE_CENTER_IDS = [362, 263, 386, 374, 373, 380]
    _NOSE_TIP_IDX = 1

    @staticmethod
    def _get_point(lm: Any) -> np.ndarray:
        return np.asarray(lm, dtype=np.float64)[:3]

    @classmethod
    def build(cls, landmarks: Any) -> np.ndarray | None:
        if landmarks is None:
            return None
        n = len(landmarks)
        if n < cls._MIN_LANDMARKS:
            logger.debug("Too few landmarks (%d) for 3D embedding", n)
            return None

        pts = np.array([cls._get_point(landmarks[i]) for i in range(n)], dtype=np.float64)
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid

        if n > max(max(cls._RIGHT_EYE_CENTER_IDS), max(cls._LEFT_EYE_CENTER_IDS)):
            left_center = np.mean([pts[i] for i in cls._LEFT_EYE_CENTER_IDS if i < n], axis=0)
            right_center = np.mean([pts[i] for i in cls._RIGHT_EYE_CENTER_IDS if i < n], axis=0)
            scale = np.linalg.norm(left_center - right_center)
        else:
            scale = np.linalg.norm(pts[cls._NOSE_TIP_IDX]) if cls._NOSE_TIP_IDX < n else 1.0

        if scale < 1e-8:
            scale = 1.0
        pts = pts / scale

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        vec = np.asarray(pcd.points, dtype=np.float64).ravel().astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-10
        return vec / norm


def build_3d_embedding(landmarks: Any) -> np.ndarray | None:
    """Public entry point (same name as legacy module for drop-in import)."""
    return FaceEmbeddingOpen3DBuilder.build(landmarks)


# Backward-compatible alias for code / docs that referenced the old class name
FaceEmbedding3DBuilder = FaceEmbeddingOpen3DBuilder
