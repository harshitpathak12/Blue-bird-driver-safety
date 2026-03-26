from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from database.alert_repository import insert_alert
from database.driver_repository import get_driver_by_id
from database.session_repository import create_session, end_session
from src.driver_identity import match_embedding_to_driver
from src.face_embedding_open3d import build_3d_embedding
from src.fusion import FusionEngine, ModelOutputs
from training.scripts.blink_perclos.drowsiness_model import ModelDrowsiness
from training.scripts.distraction_detection.distraction_detector import DistractionDetector
from training.scripts.face_detection.face_detection import FaceDetector
from training.scripts.fatigue_detection.fatigue_detection_model import ModelFatigue
from utils.logger import get_logger
from utils.overlay import OverlayRenderer

logger = get_logger(__name__)

# Frame / pipeline constants (used by both streaming and HUD overlay).
STREAM_WIDTH, STREAM_HEIGHT = 640, 480
JPEG_SEND_QUALITY = 90
MODEL_INTERVAL = 1


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to Python-native types for json.dumps."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _do_insert_alert(driver_id: str, session_id: Optional[str], alert_type: str, confidence_score: float) -> None:
    """Background insert for driver alerts (DB side-effect)."""
    try:
        insert_alert(
            driver_id=driver_id,
            session_id=session_id,
            alert_type=alert_type,
            confidence_score=confidence_score,
            gps_latitude=None,
            gps_longitude=None,
        )
        logger.debug(
            "Alert inserted: driver=%s type=%s conf=%.2f",
            driver_id,
            alert_type,
            confidence_score,
        )
    except Exception as e:
        logger.error("insert_alert failed: %s", e, exc_info=True)


class RealtimeFramePipeline:
    """Real-time frame ingestion and processing pipeline."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stream")
        self.placeholder_jpeg: Optional[bytes] = None
        self.face_detector = FaceDetector()
        self.fatigue_model = ModelFatigue()
        self.drowsiness_model = ModelDrowsiness()
        self.distraction_detector = DistractionDetector()
        self.fusion_engine = FusionEngine()
        self.overlay_renderer = OverlayRenderer()

        # Optional ArcFace embedding (falls back cleanly if torch/weights unavailable).
        try:
            from training.scripts.face_recongnition.face_recognition import ArcFaceModel

            self.arcface_model: Optional[Any] = ArcFaceModel()
            logger.info("ArcFace model loaded successfully")
        except Exception as e:
            logger.warning("ArcFace not loaded (will skip 2D recognition): %s", e)
            self.arcface_model = None

        logger.info("RealtimeFramePipeline initialised (workers=%d)", self.executor._max_workers)

    def _get_placeholder_jpeg(self) -> bytes:
        if self.placeholder_jpeg is None:
            black = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
            ok, buf = cv2.imencode(
                ".jpg",
                black,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_SEND_QUALITY],
            )
            self.placeholder_jpeg = buf.tobytes() if ok else black.tobytes()
        return self.placeholder_jpeg

    def process_frame(
        self,
        data: bytes,
        run_models: bool,
        frame_count: int,
        driver_id: Optional[str],
        recognition_result: dict,
        last_metrics: dict,
    ) -> tuple:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None, None, last_metrics

        h_in, w_in = frame.shape[:2]
        if (w_in, h_in) != (STREAM_WIDTH, STREAM_HEIGHT):
            frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)

        landmarks, img_w, img_h = self.face_detector.get_landmarks(frame)
        has_landmarks = landmarks is not None

        if run_models:
            if has_landmarks:
                self.fatigue_model.process(frame, landmarks, img_w, img_h)
                self.drowsiness_model.process(frame, landmarks, img_w, img_h)
            else:
                self.fatigue_model.process(frame, None, None, None)
                self.drowsiness_model.process(frame, None, None, None)

            ear = self.fatigue_model.last_ear
            distraction_metrics = self.distraction_detector.process(
                frame, landmarks, img_w, img_h, ear=ear,
            )

            # Face recognition runs periodically to keep latency reasonable.
            if frame_count % 20 == 0:
                emb_3d = build_3d_embedding(landmarks) if has_landmarks else None
                emb_2d = None
                if self.arcface_model is not None:
                    try:
                        emb_2d = self.arcface_model.get_embedding_from_frame(frame)
                    except Exception as e:
                        logger.debug("ArcFace embedding extraction failed: %s", e)

                if emb_3d is not None or emb_2d is not None:
                    try:
                        driver, _ = match_embedding_to_driver(
                            embedding_2d=emb_2d,
                            embedding_3d=emb_3d,
                            driver_id=driver_id,
                        )
                        if driver is not None:
                            recognition_result["driver_id"] = driver["driver_id"]
                            recognition_result["display"] = f"{driver.get('name', driver['driver_id'])} ({driver['driver_id']})"
                        else:
                            recognition_result["driver_id"] = None
                            recognition_result["display"] = "Unknown"
                    except Exception as e:
                        logger.debug("Face recognition match error: %s", e)

            fatigue_score = 1.0 if self.fatigue_model.fatigue_active else 0.0
            perclos = self.drowsiness_model.perclos
            eye_closure_duration_sec = self.drowsiness_model.eye_closure_duration_sec

            is_distracted = distraction_metrics.get("is_distracted", False)
            distraction_duration = distraction_metrics.get("distraction_duration_sec", 0.0)

            outputs = ModelOutputs(
                perclos=perclos,
                blink_duration_sec=0.0,
                blink_rate_low=self.drowsiness_model.blink_rate_low,
                fatigue_score=fatigue_score,
                head_turned_away_sec=distraction_duration,
                distraction_score=1.0 if is_distracted else 0.0,
                eye_closure_duration_sec=eye_closure_duration_sec,
            )
            fusion_result = self.fusion_engine.fuse(outputs)

            distraction_alert = distraction_metrics.get("alert_type")
            if distraction_alert:
                effective_alert = distraction_alert
                effective_state = "distraction"
                effective_confidence = distraction_metrics.get("confidence_score", 0.8)
                effective_message = distraction_metrics.get("alert_message", "")
            else:
                effective_alert = fusion_result.alert_type
                effective_state = fusion_result.driver_state
                effective_confidence = fusion_result.confidence_score
                effective_message = fusion_result.message or ""

            recognized_display = recognition_result.get("display", "—")
            last_metrics = {
                "driver_state": effective_state,
                "dl_number": recognition_result.get("dl_number", "—"),
                "alert_type": effective_alert,
                "alert_message": effective_message,
                "confidence_score": round(effective_confidence, 3),
                "ear": round(self.fatigue_model.last_ear, 4),
                "mar": round(self.fatigue_model.last_mar, 4),
                "fatigue_active": self.fatigue_model.fatigue_active,
                "perclos": round(perclos, 4),
                "blink_count": self.drowsiness_model.blink_count,
                "blink_rate_hz": round(self.drowsiness_model.blink_rate_hz, 3),
                "blink_rate_low": self.drowsiness_model.blink_rate_low,
                "eye_closure_duration_sec": round(eye_closure_duration_sec, 2),
                "pitch": distraction_metrics.get("raw_pitch", 0.0),
                "yaw": distraction_metrics.get("raw_yaw", 0.0),
                "roll": distraction_metrics.get("raw_roll", 0.0),
                "driver_identity": recognized_display,
                "attention_state": distraction_metrics.get("attention_state", "attentive"),
                "alignment_score": distraction_metrics.get("alignment_score", 1.0),
                "head_deviation_yaw": distraction_metrics.get("head_deviation_yaw", 0.0),
                "head_deviation_pitch": distraction_metrics.get("head_deviation_pitch", 0.0),
                "gaze_deviation_h": distraction_metrics.get("gaze_deviation_h", 0.0),
                "gaze_deviation_v": distraction_metrics.get("gaze_deviation_v", 0.0),
                "distraction_duration_sec": distraction_metrics.get("distraction_duration_sec", 0.0),
                "calibrated": distraction_metrics.get("calibrated", False),
                "calibration_remaining_sec": distraction_metrics.get("calibration_remaining_sec", 0.0),
                "votes": distraction_metrics.get("votes", {}),
                "is_distracted": is_distracted,
            }

            if frame_count % 30 == 0:
                logger.debug(
                    "Frame #%d metrics: state=%s ear=%.3f perclos=%.2f attn=%s distracted=%s",
                    frame_count,
                    effective_state,
                    last_metrics["ear"],
                    last_metrics["perclos"],
                    last_metrics["attention_state"],
                    is_distracted,
                )

        composite = self.overlay_renderer.draw_driver_hud(
            frame,
            landmarks=landmarks,
            img_w=img_w,
            img_h=img_h,
            **{k: v for k, v in last_metrics.items() if k not in ("landmarks", "img_w", "img_h")},
        )

        ok, buf = cv2.imencode(
            ".jpg",
            composite,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_SEND_QUALITY],
        )
        jpeg_bytes = buf.tobytes() if ok else None
        return jpeg_bytes, last_metrics.get("alert_type"), last_metrics

    async def stream(self, websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("WebSocket client connected from %s", websocket.client)

        driver_id = websocket.query_params.get("driver_id")
        session_id: Optional[str] = None
        recognition_result: dict = {"driver_id": None, "display": "—", "dl_number": "—"}

        if driver_id:
            doc = get_driver_by_id(driver_id)
            if doc and doc.get("dl_number"):
                recognition_result["dl_number"] = str(doc["dl_number"])

        frame_count = 0
        last_metrics: dict = {
            "driver_state": "waiting",
            "dl_number": recognition_result.get("dl_number", "—"),
        }
        latest_data: Optional[bytes] = None
        loop = asyncio.get_running_loop()

        async def _recv_loop():
            nonlocal latest_data
            try:
                while True:
                    latest_data = await websocket.receive_bytes()
            except WebSocketDisconnect:
                pass

        recv_task = asyncio.create_task(_recv_loop())
        try:
            while not recv_task.done():
                if latest_data is None:
                    await asyncio.sleep(0.005)
                    continue

                data = latest_data
                latest_data = None
                run_models = (frame_count % MODEL_INTERVAL) == 0

                try:
                    jpeg_bytes, alert_type, last_metrics = await loop.run_in_executor(
                        self.executor,
                        self.process_frame,
                        data,
                        run_models,
                        frame_count,
                        driver_id,
                        recognition_result,
                        last_metrics,
                    )
                except Exception as e:
                    logger.error("Frame processing error (frame #%d): %s", frame_count, e, exc_info=True)
                    jpeg_bytes = None
                    alert_type = None

                frame_count += 1
                if jpeg_bytes is None:
                    jpeg_bytes = self._get_placeholder_jpeg()

                if alert_type and run_models:
                    effective_driver_id = driver_id or recognition_result.get("driver_id")
                    if effective_driver_id and session_id is None:
                        session_id = create_session(effective_driver_id)
                        logger.info(
                            "WebSocket session started: %s (driver=%s)",
                            session_id,
                            effective_driver_id,
                        )

                    loop.run_in_executor(
                        self.executor,
                        _do_insert_alert,
                        effective_driver_id or "UNKNOWN",
                        session_id,
                        alert_type,
                        last_metrics.get("confidence_score", 0),
                    )

                try:
                    await websocket.send_text(json.dumps(_sanitize_for_json(last_metrics)))
                    await websocket.send_bytes(jpeg_bytes)
                except Exception as e:
                    logger.error("WebSocket send failed: %s", e, exc_info=True)
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error("WebSocket stream error: %s", e, exc_info=True)
        finally:
            recv_task.cancel()
            logger.info(
                "WebSocket client disconnected (frames=%d, session=%s)",
                frame_count,
                session_id,
            )
            if session_id:
                end_session(session_id)


# Backwards-compatible module-level pipeline instance.
pipeline = RealtimeFramePipeline()

