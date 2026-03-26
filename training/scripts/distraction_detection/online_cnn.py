"""Online CNN head — frozen MobileNet-V3 backbone + trainable MLP with replay buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision.transforms import functional as TF
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch/torchvision not available — online CNN disabled")


@dataclass
class Prediction:
    label: str
    confidence: float


class HybridDriverModel:

    def __init__(
        self, device: Optional[str] = None, lr: float = 3e-4,
        train_every_n: int = 6, steps_per_update: int = 2,
        max_buffer: int = 256, min_total_samples: int = 80, min_per_class: int = 20,
    ):
        self.enabled = False
        self._frame_count = 0
        self._train_every_n = int(train_every_n)
        self._steps_per_update = int(steps_per_update)
        self._max_buffer = int(max_buffer)
        self._min_total_samples = int(min_total_samples)
        self._min_per_class = int(min_per_class)
        self._buf_x: list = []
        self._buf_y: list[int] = []

        if not TORCH_AVAILABLE:
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_large(weights=weights)
        backbone.classifier = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone.to(self.device).eval()
        self.weights = weights

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            feat_dim = self.backbone(dummy).shape[-1]

        self.geom_dim = 6
        self.total_dim = feat_dim + self.geom_dim

        self.model = nn.Sequential(
            nn.Linear(self.total_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 2),
        ).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.enabled = True

    def _counts(self) -> tuple:
        n0 = sum(1 for y in self._buf_y if y == 0)
        n1 = sum(1 for y in self._buf_y if y == 1)
        return n0, n1

    def is_ready(self) -> bool:
        n0, n1 = self._counts()
        return (n0 + n1) >= self._min_total_samples and n0 >= self._min_per_class and n1 >= self._min_per_class

    def _preprocess(self, roi_bgr: np.ndarray):
        rgb = roi_bgr[:, :, ::-1].copy()
        t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        t = TF.resize(t, [224, 224], antialias=True)
        t = TF.normalize(t, mean=list(self.weights.transforms().mean), std=list(self.weights.transforms().std))
        return t.unsqueeze(0)

    def extract_features(self, roi_bgr, geom_features):
        x = self._preprocess(roi_bgr)
        with torch.no_grad():
            cnn_feat = self.backbone(x).flatten()
        geom_t = torch.as_tensor(geom_features, device=self.device, dtype=torch.float32).flatten()
        return torch.cat([cnn_feat, geom_t], dim=0)

    def predict(self, roi_bgr, geom_features) -> Prediction:
        if not self.enabled:
            return Prediction("unknown", 0.0)
        if not self.is_ready():
            return Prediction("warming_up", 0.0)
        feat = self.extract_features(roi_bgr, geom_features).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(feat)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
        return Prediction("normal" if pred == 0 else "distraction", float(probs[pred]))

    def train_online(self, roi_bgr, geom_features, teacher_label: Optional[int] = None) -> None:
        if not self.enabled or teacher_label not in (0, 1):
            return
        feat = self.extract_features(roi_bgr, geom_features).detach()
        self._buf_x.append(feat)
        self._buf_y.append(int(teacher_label))
        if len(self._buf_x) > self._max_buffer:
            self._buf_x = self._buf_x[-self._max_buffer:]
            self._buf_y = self._buf_y[-self._max_buffer:]

        self._frame_count += 1
        if self._frame_count % self._train_every_n != 0:
            return

        idx0 = [i for i, y in enumerate(self._buf_y) if y == 0]
        idx1 = [i for i, y in enumerate(self._buf_y) if y == 1]
        if not idx0 or not idx1:
            return

        self.model.train()
        for _ in range(self._steps_per_update):
            y_target = int(np.random.randint(0, 2))
            pool = idx0 if y_target == 0 else idx1
            i = int(np.random.choice(pool))
            x = self._buf_x[i].unsqueeze(0)
            y = torch.tensor([self._buf_y[i]], device=self.device, dtype=torch.long)
            loss = self.criterion(self.model(x), y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.model.eval()

    def reset(self):
        self._buf_x.clear()
        self._buf_y.clear()
        self._frame_count = 0
