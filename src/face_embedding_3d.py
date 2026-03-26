"""
Backward-compatible module path for 3D face embeddings.

Implementation was moved to `src.face_embedding_open3d` (Open3D). Import from there for new code.
"""

from .face_embedding_open3d import (
    FaceEmbedding3DBuilder,
    FaceEmbeddingOpen3DBuilder,
    build_3d_embedding,
)

__all__ = [
    "build_3d_embedding",
    "FaceEmbedding3DBuilder",
    "FaceEmbeddingOpen3DBuilder",
]
