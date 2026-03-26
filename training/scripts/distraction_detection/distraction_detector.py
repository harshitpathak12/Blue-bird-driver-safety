"""Unified distraction detector — calibration-based voting system.

Wraps head pose estimation, eye gaze estimation, attention model,
geometric temporal classifier, LSTM temporal model, and online CNN
into a single process() call that returns a comprehensive metrics dict.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

from training.scripts.distraction_detection.attention_model import (
    AttentionModel,
    AttentionState,
)
from training.scripts.distraction_detection.eye_gaze_estimator import EyeGazeEstimator
from training.scripts.distraction_detection.face_3d_features import extract_face_3d_features
from training.scripts.distraction_detection.geometric_temporal_classifier import (
    GeometricTemporalClassifier,
    GeometricTemporalState,
)
from training.scripts.distraction_detection.head_pose_estimator import HeadPoseEstimator
from training.scripts.distraction_detection.temporal_attention_model import (
    TemporalAttentionModel,
    TemporalAttentionState,
    build_frame_feature_vector,
)

try:
    from training.scripts.distraction_detection.online_cnn import HybridDriverModel
    _CNN_AVAILABLE = True
except Exception:
    _CNN_AVAILABLE = False

CNN_ONLINE_ENABLED = True
CNN_DISTRACT_CONF_ON = 0.70
CNN_DISTRACT_CONF_OFF = 0.55
CNN_SUSTAIN_FRAMES = 8
CNN_VOTE_MIN_CONF = 0.80
LSTM_VOTE_MIN_CONF = 0.75
CNN_WARMUP_TRAIN_SEC = 30.0


class DistractionDetector:
    """Unified distraction detection with voting (rule-based + LSTM + CNN)."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="distract")
        self.head_pose_estimator = HeadPoseEstimator()
        self.eye_gaze_estimator = EyeGazeEstimator()
        self.attention_model = AttentionModel()
        self.geometric_classifier = GeometricTemporalClassifier()
        self.temporal_lstm = TemporalAttentionModel()

        self.online_cnn: Optional[Any] = None
        if CNN_ONLINE_ENABLED and _CNN_AVAILABLE:
            try:
                self.online_cnn = HybridDriverModel()
                logger.info("Online CNN (HybridDriverModel) loaded")
            except Exception as e:
                logger.warning("Online CNN not available: %s", e)

        self._cnn_off_latched = False
        self._cnn_consecutive = 0
        self._cnn_train_start_ts: Optional[float] = None

        self.last_pitch: float = 0.0
        self.last_yaw: float = 0.0
        self.last_roll: float = 0.0

        logger.info("DistractionDetector initialised (cnn=%s, lstm=%s)", self.online_cnn is not None, True)

    def recalibrate(self) -> None:
        logger.info("Recalibrating distraction detector")
        self.attention_model.recalibrate()
        self.geometric_classifier.clear_buffer()
        self.temporal_lstm.clear_buffer()
        self._cnn_off_latched = False
        self._cnn_consecutive = 0
        self._cnn_train_start_ts = None
        if self.online_cnn is not None and hasattr(self.online_cnn, "reset"):
            self.online_cnn.reset()

    def _face_roi_from_landmarks(
        self, frame: np.ndarray, landmarks: List[np.ndarray], w: int, h: int,
    ) -> Optional[np.ndarray]:
        try:
            pts = np.array([[lm[0] * w, lm[1] * h] for lm in landmarks], dtype=np.float32)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            mx = 0.15 * (x1 - x0)
            my = 0.20 * (y1 - y0)
            x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
            x1, y1 = int(min(w - 1, x1 + mx)), int(min(h - 1, y1 + my))
            if x1 <= x0 or y1 <= y0:
                return None
            roi = frame[y0:y1, x0:x1]
            return roi if roi.size > 0 else None
        except Exception:
            return None

    def process(
        self,
        frame: np.ndarray,
        landmarks: Optional[List[np.ndarray]],
        img_w: Optional[int],
        img_h: Optional[int],
        ear: float = 0.0,
    ) -> Dict[str, Any]:
        """Process one frame and return distraction metrics dict."""
        if landmarks is None or img_w is None or img_h is None:
            return self._no_face_metrics()

        pose_fut = self._executor.submit(self.head_pose_estimator.process, landmarks, img_w, img_h)
        gaze_fut = self._executor.submit(self.eye_gaze_estimator.process, landmarks, img_w, img_h)
        pose = pose_fut.result()
        gaze = gaze_fut.result()

        raw_pitch, raw_yaw, raw_roll = (pose if pose is not None else (0.0, 0.0, 0.0))
        gaze_h, gaze_v = (gaze if gaze is not None else (0.5, 0.5))

        self.last_pitch, self.last_yaw, self.last_roll = raw_pitch, raw_yaw, raw_roll

        attn_res = self.attention_model.process(
            head_pitch=raw_pitch, head_yaw=raw_yaw,
            gaze_avg_h=gaze_h, gaze_avg_v=gaze_v,
        )

        if attn_res.state != AttentionState.CALIBRATING:
            self.geometric_classifier.push(
                attn_res.head_deviation_yaw, attn_res.head_deviation_pitch,
                attn_res.gaze_deviation_h, attn_res.gaze_deviation_v,
                attn_res.alignment_score,
            )
        t_state, t_conf = self.geometric_classifier.predict()

        if self.attention_model.calibrated and self._cnn_train_start_ts is None:
            self._cnn_train_start_ts = time.time()
        cnn_training_remaining = 0.0
        if self._cnn_train_start_ts is not None:
            cnn_training_remaining = max(0.0, CNN_WARMUP_TRAIN_SEC - (time.time() - self._cnn_train_start_ts))

        lstm_state = TemporalAttentionState.WARMING_UP
        lstm_conf = 0.0
        lstm_vote: Optional[str] = None
        if attn_res.state != AttentionState.CALIBRATING:
            face_3d = extract_face_3d_features(landmarks)
            frame_vec = build_frame_feature_vector(
                head_dev_yaw=float(attn_res.head_deviation_yaw),
                head_dev_pitch=float(attn_res.head_deviation_pitch),
                gaze_dev_h=float(attn_res.gaze_deviation_h),
                gaze_dev_v=float(attn_res.gaze_deviation_v),
                alignment_score=float(attn_res.alignment_score),
                ear=float(ear),
                face_3d_vec=face_3d,
            )
            self.temporal_lstm.push_features(frame_vec)
            lstm_state, lstm_conf = self.temporal_lstm.predict()
            if lstm_state == TemporalAttentionState.DISTRACTED:
                lstm_vote = "distraction"
            elif lstm_state == TemporalAttentionState.ATTENTIVE:
                lstm_vote = "normal"

        cnn_label, cnn_conf, cnn_driver_state = None, 0.0, None
        if self.online_cnn is not None and getattr(self.online_cnn, "enabled", False):
            roi = self._face_roi_from_landmarks(frame, landmarks, img_w, img_h)
            if roi is not None and attn_res.state != AttentionState.CALIBRATING:
                try:
                    geom_features = np.array(
                        [raw_yaw, raw_pitch, raw_roll, (gaze_h - 0.5), (gaze_v - 0.5), ear],
                        dtype=np.float32,
                    )
                    teacher_label = None
                    if lstm_vote == "normal" and lstm_conf >= LSTM_VOTE_MIN_CONF:
                        teacher_label = 0
                    elif lstm_vote == "distraction" and lstm_conf >= LSTM_VOTE_MIN_CONF:
                        teacher_label = 1
                    self.online_cnn.train_online(roi, geom_features, teacher_label=teacher_label)
                    pred = self.online_cnn.predict(roi, geom_features)
                    cnn_label = str(pred.label)
                    cnn_conf = float(pred.confidence)
                except Exception:
                    cnn_label, cnn_conf = None, 0.0

                if cnn_label in ("warming_up", "unknown", None):
                    self._cnn_consecutive = max(0, self._cnn_consecutive - 1)
                    self._cnn_off_latched = False
                    cnn_driver_state = "normal"
                elif cnn_label == "distraction":
                    if not self._cnn_off_latched and cnn_conf >= CNN_DISTRACT_CONF_ON:
                        self._cnn_off_latched = True
                        self._cnn_consecutive = 0
                    if self._cnn_off_latched:
                        self._cnn_consecutive += 1
                elif cnn_label == "normal":
                    if self._cnn_off_latched and cnn_conf >= CNN_DISTRACT_CONF_OFF:
                        self._cnn_consecutive = max(0, self._cnn_consecutive - 2)
                        if self._cnn_consecutive == 0:
                            self._cnn_off_latched = False
                    else:
                        self._cnn_consecutive = max(0, self._cnn_consecutive - 1)

                if cnn_driver_state is None:
                    cnn_driver_state = "distraction" if (self._cnn_off_latched and self._cnn_consecutive >= CNN_SUSTAIN_FRAMES) else "normal"

        geom_vote = "distraction" if (t_state == GeometricTemporalState.DISTRACTED or attn_res.state == AttentionState.DISTRACTED) else "normal"

        cnn_vote: Optional[str] = None
        if cnn_driver_state in ("normal", "distraction"):
            cnn_vote = ("distraction" if cnn_conf >= CNN_VOTE_MIN_CONF else None) if cnn_driver_state == "distraction" else "normal"

        lstm_vote_gated: Optional[str] = None
        if lstm_vote in ("normal", "distraction"):
            lstm_vote_gated = ("distraction" if lstm_conf >= LSTM_VOTE_MIN_CONF else None) if lstm_vote == "distraction" else "normal"

        votes: list[str] = [geom_vote]
        if cnn_vote is not None and cnn_training_remaining <= 0.0:
            votes.append(cnn_vote)
        if lstm_vote_gated is not None:
            votes.append(lstm_vote_gated)

        distraction_votes = sum(1 for v in votes if v == "distraction")
        normal_votes = sum(1 for v in votes if v == "normal")
        final_state = "distraction" if distraction_votes > normal_votes else "normal"

        logger.debug(
            "Votes: rule=%s cnn=%s lstm=%s → %s (dist=%d norm=%d)",
            geom_vote, cnn_vote, lstm_vote_gated, final_state, distraction_votes, normal_votes,
        )

        if attn_res.state == AttentionState.CALIBRATING:
            driver_state = "calibrating"
            alert_type = None
            alert_message = "Look forward for 5 seconds to set baseline."
            confidence_score = 0.0
            final_state = "normal"
        else:
            driver_state = final_state
            if final_state == "distraction":
                alert_type = "distraction"
                alert_message = "Eyes off the road! Please refocus."
                confidence_score = float(max(t_conf, cnn_conf, lstm_conf))
            else:
                alert_type = None
                alert_message = ""
                confidence_score = 0.0

        return {
            "driver_state": driver_state,
            "final_state": final_state,
            "is_distracted": final_state == "distraction",
            "attention_state": attn_res.state.value,
            "alignment_score": round(attn_res.alignment_score, 3),
            "head_deviation_yaw": round(attn_res.head_deviation_yaw, 2),
            "head_deviation_pitch": round(attn_res.head_deviation_pitch, 2),
            "gaze_deviation_h": round(attn_res.gaze_deviation_h, 4),
            "gaze_deviation_v": round(attn_res.gaze_deviation_v, 4),
            "distraction_duration_sec": round(attn_res.distraction_duration_sec, 2),
            "raw_pitch": round(raw_pitch, 2),
            "raw_yaw": round(raw_yaw, 2),
            "raw_roll": round(raw_roll, 2),
            "calibrated": attn_res.calibrated,
            "calibration_remaining_sec": round(getattr(attn_res, "calibration_remaining_sec", 0.0), 1),
            "temporal_state": t_state.value,
            "temporal_confidence": round(t_conf, 3),
            "lstm_state": lstm_state.value,
            "lstm_confidence": round(float(lstm_conf), 3),
            "alert_type": alert_type,
            "alert_message": alert_message,
            "confidence_score": round(confidence_score, 3),
            "votes": {
                "rule_based": geom_vote,
                "cnn": (cnn_vote if cnn_vote in ("normal", "distraction") else "normal"),
                "lstm": (lstm_vote_gated if lstm_vote_gated in ("normal", "distraction") else "normal"),
                "final": final_state,
            },
        }

    def _no_face_metrics(self) -> Dict[str, Any]:
        return {
            "driver_state": "no_face",
            "final_state": "no_face",
            "is_distracted": False,
            "attention_state": AttentionState.CALIBRATING.value if not self.attention_model.calibrated else "attentive",
            "alignment_score": 1.0,
            "head_deviation_yaw": 0.0,
            "head_deviation_pitch": 0.0,
            "gaze_deviation_h": 0.0,
            "gaze_deviation_v": 0.0,
            "distraction_duration_sec": 0.0,
            "raw_pitch": 0.0,
            "raw_yaw": 0.0,
            "raw_roll": 0.0,
            "calibrated": self.attention_model.calibrated,
            "calibration_remaining_sec": 5.0,
            "temporal_state": GeometricTemporalState.WARMING_UP.value,
            "temporal_confidence": 0.0,
            "lstm_state": TemporalAttentionState.WARMING_UP.value,
            "lstm_confidence": 0.0,
            "alert_type": None,
            "alert_message": "",
            "confidence_score": 0.0,
            "votes": {"rule_based": "normal", "cnn": "normal", "lstm": "normal", "final": "normal"},
        }
