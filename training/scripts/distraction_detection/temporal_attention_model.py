"""Temporal attention model: LSTM over sequences of 3D face + geometric features."""

from __future__ import annotations

import os
from collections import deque
from enum import Enum
from typing import Deque, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — LSTM temporal model disabled")

GEOMETRIC_DIM = 6
FACE_3D_DIM = 72
FEATURE_DIM = GEOMETRIC_DIM + FACE_3D_DIM
SEQ_LEN = 30
NUM_CLASSES = 2

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")


class TemporalAttentionState(Enum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    WARMING_UP = "warming_up"


if TORCH_AVAILABLE:
    class TemporalAttentionLSTM(nn.Module):
        def __init__(self, input_size=FEATURE_DIM, hidden_size=64, num_layers=1, num_classes=NUM_CLASSES, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(32, num_classes))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


class TemporalAttentionModel:
    """Runtime wrapper for LSTM temporal attention model."""

    def __init__(self, seq_len: int = SEQ_LEN, model_path: Optional[str] = None, device: Optional[str] = None):
        self.seq_len = seq_len
        self._buffer: Deque[np.ndarray] = deque(maxlen=seq_len)
        self._model = None
        self._device = device
        self._enabled = False

        if not TORCH_AVAILABLE:
            return

        if model_path is None:
            model_dir = os.environ.get("MODEL_BASE_DIR", _DEFAULT_MODEL_DIR)
            model_path = os.path.join(model_dir, "temporal_attention_lstm.pth")

        if os.path.exists(model_path):
            try:
                self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self._model = TemporalAttentionLSTM()
                ckpt = torch.load(model_path, map_location=self._device, weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                self._model.load_state_dict(state)
                self._model.to(self._device)
                self._model.eval()
                self._enabled = True
                logger.info("TemporalAttentionLSTM loaded from %s (device=%s)", model_path, self._device)
            except Exception as exc:
                logger.error("TemporalAttentionLSTM failed to load %s: %s", model_path, exc, exc_info=True)
        else:
            logger.warning("No LSTM checkpoint at %s — using rule-based fallback", model_path)

    def push_features(self, feature_vec: np.ndarray) -> None:
        if feature_vec.size != FEATURE_DIM:
            return
        self._buffer.append(feature_vec.copy())

    def predict(self) -> Tuple[TemporalAttentionState, float]:
        if not self._enabled or self._model is None or len(self._buffer) < self.seq_len:
            return TemporalAttentionState.WARMING_UP, 0.0

        arr = np.stack(list(self._buffer), axis=0).astype(np.float32)
        t = torch.from_numpy(arr).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        state = TemporalAttentionState.DISTRACTED if pred == 1 else TemporalAttentionState.ATTENTIVE
        return state, conf

    @property
    def enabled(self) -> bool:
        return self._enabled

    def clear_buffer(self) -> None:
        self._buffer.clear()


def build_frame_feature_vector(
    head_dev_yaw: float, head_dev_pitch: float,
    gaze_dev_h: float, gaze_dev_v: float,
    alignment_score: float, ear: float,
    face_3d_vec: Optional[np.ndarray],
) -> np.ndarray:
    geom = np.array([head_dev_yaw, head_dev_pitch, gaze_dev_h, gaze_dev_v, alignment_score, ear], dtype=np.float32)
    if face_3d_vec is not None and face_3d_vec.size == FACE_3D_DIM:
        return np.concatenate([geom, face_3d_vec]).astype(np.float32)
    return np.concatenate([geom, np.zeros(FACE_3D_DIM, dtype=np.float32)]).astype(np.float32)
