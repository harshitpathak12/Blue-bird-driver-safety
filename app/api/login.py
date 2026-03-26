"""
/api/login – Driver authentication (PDF: Face Detection + Recognition).
Login: face image → match (2D and/or 3D) → driver_id, driver_name, age.
Register: live photo (base64) + name, age → MediaPipe landmarks → Open3D PointCloud 3D embedding → face_embedding_3d.
verify-dl: after registration, image of driving licence → YOLO + Qwen OCR + rules; name/age must match DB.

Strict registration + DL (used by ``data_pipeline.client`` with external DL WebSocket + ``POST /finalize-dl``):
    ``POST /register`` stores a **pending** record in **process memory** (embedding + name + age), returns ``pending_id``.
    The local client streams frames to the **driver_license_detection**-style service; on ``valid`` it calls
    ``POST /finalize-dl`` with OCR fields so the server runs ``registration_matches_dl`` and performs the **first**
    MongoDB write (``create_driver`` + ``update_driving_license``).

Deployment: pending state is **not** shared across processes. Run **one** uvicorn worker (or gunicorn ``-w 1``),
or use a shared store (Redis) if you scale horizontally — otherwise ``finalize-dl`` can return
"Pending registration expired or not found" after a successful ``/register`` on another worker.
"""

import asyncio
import base64
import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image

from src.schemas.payloads import LoginResponse, RegisterLiveBody, RegisterResponse
from src.driver_identity import match_embedding_to_driver
from src.face_embedding_open3d import build_3d_embedding
from database import driver_repository
from database.driver_repository import DriverRepository
from training.scripts.face_detection.face_detection import FaceDetector
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/login", tags=["login"])

_dl_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dl_verify")
_dl_stream_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dl_stream")

_PENDING_TTL_SEC = int(os.environ.get("DL_PENDING_TTL_SEC", "300"))


class PendingRegistration:
    def __init__(self, *, name: str, age: Optional[int], face_embedding_3d: list[float]):
        self.name = name
        self.age = age
        self.face_embedding_3d = face_embedding_3d
        self.created_at = time.time()


_pending_lock = threading.Lock()
_pending_registrations: dict[str, PendingRegistration] = {}


def _new_pending_id() -> str:
    # Short id for compact form payloads (still collision-resistant enough).
    return uuid.uuid4().hex[:10].upper()


def _pending_get(pending_id: str) -> PendingRegistration | None:
    now = time.time()
    with _pending_lock:
        pending = _pending_registrations.get(pending_id)
        if not pending:
            return None
        if now - pending.created_at > _PENDING_TTL_SEC:
            _pending_registrations.pop(pending_id, None)
            return None
        return pending


def _pending_delete(pending_id: str) -> None:
    with _pending_lock:
        _pending_registrations.pop(pending_id, None)


def shutdown_dl_verification_executor() -> None:
    """Wait for in-flight DL verification work; call on application shutdown."""
    _dl_executor.shutdown(wait=True)
    _dl_stream_executor.shutdown(wait=True)

try:
    # Lazy import: training/scripts/...face_recognition imports RetinaFace/TensorFlow.
    # This keeps ArcFace as an optional fallback, instead of breaking server startup
    # when TensorFlow/RetinaFace is not installed on the VM.
    from training.scripts.face_recongnition.face_recognition import ArcFaceModel

    _arcface_model: Any | None = ArcFaceModel()
    logger.info("ArcFace model ready for login route")
except Exception as e:
    logger.warning("ArcFace model not available for login: %s", e)
    _arcface_model = None

_face_detector: FaceDetector | None = FaceDetector()


def _get_face_model() -> Any | None:
    return _arcface_model


def _coerce_age_int(value: Any) -> Optional[int]:
    """Normalize Mongo / JSON age to int or None (same interpretation as before)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else None
    return None


def _decode_image(raw: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to BGR frame. Tries OpenCV first, then Pillow for broader format support."""
    if not raw or len(raw) == 0:
        return None
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil_img = Image.open(io.BytesIO(raw))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img = np.array(pil_img)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _get_landmarks_from_image(img: np.ndarray):
    """Run MediaPipe Face Landmarker on full image; return (landmarks, img_w, img_h) or (None, None, None)."""
    if _face_detector is None:
        return None, None, None
    return _face_detector.get_landmarks(img)


def _extract_face_from_bytes(raw: bytes):
    """Decode image bytes and return largest face crop in BGR."""
    nparr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    model = _get_face_model()
    if model is None:
        return None

    try:
        detections = model.detector.detect_faces(img)
    except Exception:
        return None
    if not detections:
        return None

    faces = list(detections.values())
    largest = max(
        faces,
        key=lambda d: (d["facial_area"][2] - d["facial_area"][0])
        * (d["facial_area"][3] - d["facial_area"][1]),
    )
    x1, y1, x2, y2 = largest["facial_area"]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    face = img[y1:y2, x1:x2]
    if face.shape[0] < 50 or face.shape[1] < 50:
        return None
    return face


@router.post("/", response_model=LoginResponse)
async def login(
    driver_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """
    Login with face image. If driver_id provided, verify against that driver; else match against DB.
    Uses 3D embedding (MediaPipe) when available and 2D (ArcFace) for legacy drivers.
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image")

    logger.info("Login attempt (driver_id=%s, image_size=%d bytes)", driver_id, len(raw))
    img = _decode_image(raw)
    if img is None:
        logger.warning("Login failed: could not decode image")
        raise HTTPException(status_code=400, detail="Could not decode image")

    emb_3d = None
    landmarks, _, _ = _get_landmarks_from_image(img)
    if landmarks is not None:
        emb_3d = build_3d_embedding(landmarks)

    emb_2d = None
    model = _get_face_model()
    if model is not None:
        face = _extract_face_from_bytes(raw)
        if face is not None:
            emb_2d = model.get_embedding(face)

    if emb_3d is None and emb_2d is None:
        raise HTTPException(status_code=400, detail="No valid face detected in image")

    driver, score = match_embedding_to_driver(
        embedding_2d=emb_2d,
        embedding_3d=emb_3d,
        driver_id=driver_id,
    )
    if not driver:
        logger.info("Login failed: face not recognised (score=%.3f)", score)
        raise HTTPException(status_code=401, detail="Face not recognized or invalid driver_id")

    logger.info("Login success: driver=%s score=%.3f", driver["driver_id"], score)
    driver_repository.update_last_seen(driver["driver_id"])
    return LoginResponse(
        driver_id=driver["driver_id"],
        driver_name=driver.get("name", ""),
        age=driver.get("age"),
        message="Login successful",
    )


def _normalize_image_base64(value: str) -> str:
    """Strip optional data URL prefix; remove newlines only (preserve space for +-fix later)."""
    s = (value or "").strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    # Remove newlines/tabs so line-wrapped base64 works; do not remove spaces (may be corrupted +)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "")
    return s


@router.post("/register", response_model=RegisterResponse)
async def register(body: RegisterLiveBody):
    """
    First-time registration: live photo only (no file upload).
    Uses MediaPipe Face Landmarker to get 3D landmarks, Open3D-based embedding, and stores face_embedding_3d.
    """
    image_b64 = _normalize_image_base64(body.image_base64)
    if not image_b64:
        raise HTTPException(
            status_code=400,
            detail="Missing image_base64. Send raw base64 or data URL (data:image/...;base64,...).",
        )
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except Exception:
        # If sent as form data, + may have been converted to space; try fixing
        try:
            raw = base64.b64decode(image_b64.replace(" ", "+"), validate=True)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image_base64: ensure raw base64 or data URL. ({e!r})",
            )

    if not raw or len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty image")

    img = _decode_image(raw)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Ensure image_base64 is valid base64-encoded JPEG or PNG (e.g. from canvas.toDataURL('image/jpeg') or file read).",
        )

    landmarks, _, _ = _get_landmarks_from_image(img)
    if landmarks is None:
        raise HTTPException(
            status_code=400,
            detail="No valid face detected. Use a live photo from the camera.",
        )

    emb_3d = build_3d_embedding(landmarks)
    if emb_3d is None:
        raise HTTPException(
            status_code=400,
            detail="Could not build face embedding. Ensure a clear face is visible.",
        )

    # Strict mode: do NOT create a MongoDB driver record yet.
    # Store it as a pending registration in memory, then create the driver
    # only after DL verification succeeds.
    pending_id = _new_pending_id()
    with _pending_lock:
        _pending_registrations[pending_id] = PendingRegistration(
            name=body.name,
            age=body.age,
            face_embedding_3d=emb_3d.astype(float).tolist(),
        )

    logger.info(
        "Registration stored as pending: pending_id=%s name=%s age=%s",
        pending_id,
        body.name,
        body.age,
    )
    return RegisterResponse(
        pending_id=pending_id,
        driver_id=None,
        driver_name=body.name,
        age=body.age,
        message="Registration successful (waiting for driving licence verification)",
    )


def _verify_dl_job(pending_id: str, img: np.ndarray) -> dict[str, Any]:
    """Runs in thread pool: YOLO + OCR + conditional DB create/update (strict mode)."""
    try:
        from training.scripts.driving_license_model.dl_matching import registration_matches_dl
        from training.scripts.driving_license_model.dl_validator import validate_license_frame
    except ImportError as e:
        logger.error("DL verification dependencies missing: %s", e, exc_info=True)
        return {
            "_http_status": 503,
            "detail": (
                "Driving licence verification is not available on this server: "
                "install PyTorch, ultralytics, transformers, and qwen-vl-utils "
                "(see requirements.txt). Import error: %s" % e
            ),
        }

    pending = _pending_get(pending_id)
    if not pending:
        return {"_http_status": 400, "detail": "Pending registration expired or not found"}

    try:
        result = validate_license_frame(img)
    except FileNotFoundError as e:
        return {"_http_status": 503, "detail": str(e)}
    except Exception as e:
        logger.error("DL validation error: %s", e, exc_info=True)
        return {"_http_status": 500, "detail": f"Driving licence check failed: {e!s}"}

    verdict = (result.get("verdict") or "").lower()
    if verdict != "valid":
        msg = result.get("message") or result.get("rule_reason") or "Invalid or unreadable driving licence"
        return {
            "_http_status": 400,
            "detail": msg,
            "verdict": verdict,
        }

    ocr_text = result.get("ocr_text") or ""
    reg_name = (pending.name or "").strip()
    reg_age = _coerce_age_int(pending.age)

    match = registration_matches_dl(reg_name, reg_age, ocr_text)
    if not match.get("ok"):
        reason = match.get("reason", "")
        _pending_delete(pending_id)
        if reason == "name_mismatch":
            return {"_http_status": 400, "detail": "Mismatched Name", "verdict": verdict}
        return {"_http_status": 400, "detail": "Mismatched Age", "verdict": verdict}

    dl_nums = result.get("dl_numbers") or []
    dl_number = str(dl_nums[0]).strip() if dl_nums else ""
    if not dl_number:
        _pending_delete(pending_id)
        return {"_http_status": 400, "detail": "Could not read DL number from licence", "verdict": verdict}

    ve = result.get("validity_end")
    holder = match.get("holder_name")

    # First DB write happens only now (strict mode).
    driver = driver_repository.create_driver(
        driver_id=None,
        name=pending.name,
        age=pending.age,
        face_embedding=None,
        face_embedding_3d=pending.face_embedding_3d,
        face_image_path=None,
    )

    DriverRepository.update_driving_license(
        driver["driver_id"],
        dl_number=dl_number,
        dl_verified=True,
        dl_name_on_card=holder,
        dl_validity_end=str(ve) if ve else None,
        registration_name_match=True,
        registration_age_match=bool(match.get("age_ok", True)),
    )

    _pending_delete(pending_id)

    return {
        "_http_status": 200,
        "driver_id": driver["driver_id"],
        "dl_number": dl_number,
        "validity_end": ve,
        "message": "Driving licence verified",
    }


@router.post("/verify-dl")
async def verify_driving_license(
    pending_id: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Verify Indian driving licence (photo) against the registered driver record.
    Requires YOLO weights under models/driving_license/best.pt and Qwen (first run downloads weights).
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image")

    img = _decode_image(raw)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    loop = asyncio.get_running_loop()
    out = await loop.run_in_executor(_dl_executor, _verify_dl_job, pending_id, img)

    status = int(out.pop("_http_status", 500))
    if status != 200:
        raise HTTPException(status_code=status, detail=out.get("detail", "Verification failed"))
    return out


def finalize_pending_registration_core(
    pending_id: str,
    verdict: str,
    dl_number: str,
    ocr_text: str,
    validity_end: Optional[str] = None,
) -> dict[str, Any]:
    """
    Shared DB finalization for POST /finalize-dl and the server-side DL WebSocket stream.

    Returns dict with ``ok: bool``. On success includes driver_id, dl_number, validity_end, message.
    On failure includes ``detail`` (for HTTP or JSON error).
    """
    pending = _pending_get(pending_id)
    if not pending:
        return {"ok": False, "detail": "Pending registration expired or not found"}

    verdict_l = (verdict or "").lower().strip()
    if verdict_l != "valid":
        _pending_delete(pending_id)
        return {"ok": False, "detail": "Invalid or unreadable driving licence"}

    dl_number_s = (dl_number or "").strip()
    if not dl_number_s:
        _pending_delete(pending_id)
        return {"ok": False, "detail": "Could not read DL number from licence"}

    from training.scripts.driving_license_model.dl_matching import registration_matches_dl

    match = registration_matches_dl(pending.name, pending.age, ocr_text or "")
    if not match.get("ok"):
        _pending_delete(pending_id)
        reason = match.get("reason", "")
        if reason == "name_mismatch":
            return {"ok": False, "detail": "Mismatched Name"}
        return {"ok": False, "detail": "Mismatched Age"}

    holder = match.get("holder_name")

    driver = driver_repository.create_driver(
        driver_id=None,
        name=pending.name,
        age=pending.age,
        face_embedding=None,
        face_embedding_3d=pending.face_embedding_3d,
        face_image_path=None,
    )

    DriverRepository.update_driving_license(
        driver["driver_id"],
        dl_number=dl_number_s,
        dl_verified=True,
        dl_name_on_card=holder,
        dl_validity_end=validity_end,
        registration_name_match=True,
        registration_age_match=bool(match.get("age_ok", True)),
    )

    _pending_delete(pending_id)

    logger.info(
        "finalize-dl OK: driver_id=%s dl_number=%s pending cleared",
        driver["driver_id"],
        dl_number_s,
    )
    return {
        "ok": True,
        "driver_id": driver["driver_id"],
        "dl_number": dl_number_s,
        "validity_end": validity_end,
        "message": "Driving licence verified",
    }


@router.post("/finalize-dl")
async def finalize_driving_license(
    pending_id: str = Form(...),
    verdict: str = Form(..., description="Expected: valid"),
    dl_number: str = Form(..., description="Extracted DL number"),
    validity_end: Optional[str] = Form(None, description="Optional validity end date (ISO or other string)"),
    ocr_text: str = Form(..., description="Full OCR text used for name/age verification"),
):
    """
    Strict finalization endpoint.

    This endpoint assumes the driving licence pipeline has already produced:
    - `verdict` (valid/invalid)
    - `dl_number`
    - `ocr_text`

    Server-side it only performs:
    - pending lookup
    - name/age match check via registration_matches_dl
    - then the *first* MongoDB write: create_driver + update_driving_license
    """
    out = finalize_pending_registration_core(
        pending_id,
        verdict=verdict,
        dl_number=dl_number,
        ocr_text=ocr_text,
        validity_end=validity_end,
    )
    if not out.get("ok"):
        raise HTTPException(status_code=400, detail=out.get("detail", "Verification failed"))
    return {
        "driver_id": out["driver_id"],
        "dl_number": out["dl_number"],
        "validity_end": out.get("validity_end"),
        "message": out.get("message", "Driving licence verified"),
    }


# --- Server-side DL streaming (same host as API; YOLO per frame + async Qwen like driver_license_detection) ---
_DL_STREAM_W = 640
_DL_STREAM_H = 480
_DL_STREAM_JPEG_Q = 50
_DL_PLACEHOLDER_JPEG: bytes | None = None
# 0 = finalize as soon as licence reads valid (name/age checked in finalize_pending_registration_core).
# Set e.g. 60 to require that many seconds on the connection before allowing finalize.
_DL_MIN_FINALIZE_SEC = float(os.environ.get("DL_VERIFY_MIN_SESSION_SEC", "0"))
# First spurious invalid (OCR flicker / glare) does not end the session; second invalid rejects.
_DL_IGNORE_FIRST_INVALID = os.environ.get("DL_VERIFY_IGNORE_FIRST_INVALID", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _dl_placeholder_jpeg(encode_params: list[int]) -> bytes:
    global _DL_PLACEHOLDER_JPEG
    if _DL_PLACEHOLDER_JPEG is None:
        black = np.zeros((_DL_STREAM_H, _DL_STREAM_W, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", black, encode_params)
        _DL_PLACEHOLDER_JPEG = buf.tobytes() if ok else black.tobytes()
    return _DL_PLACEHOLDER_JPEG


@router.websocket("/ws/dl-verify")
async def dl_verify_websocket(websocket: WebSocket) -> None:
    """
    Stream JPEG frames from the client; YOLO every frame + Qwen OCR in a background thread
    (throttled/cached, same model as ``driver_license_detection`` ``/stream``).

    When the model reports ``valid`` (with DL number and OCR text), frame streaming stops: we call
    ``finalize_pending_registration_core`` (name + age vs registration), persist the driver, then
    close the WebSocket. Optional ``DL_VERIFY_MIN_SESSION_SEC`` > 0 delays finalize until that
    many seconds have elapsed on the connection.

    The first ``invalid`` verdict is ignored (see ``DL_VERIFY_IGNORE_FIRST_INVALID``); a second
    ``invalid`` rejects and closes the connection.

    Query: ``pending_id`` (required), ``qwen=1`` (ignored; full OCR pipeline when card detected).

    Protocol matches ``driver_license_detection`` clients: binary JPEG (annotated), then JSON.
    """
    await websocket.accept()
    pending_id = (websocket.query_params.get("pending_id") or "").strip()
    if not pending_id or _pending_get(pending_id) is None:
        await websocket.close(code=1008)
        return

    loop = asyncio.get_running_loop()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), _DL_STREAM_JPEG_Q]
    t_ws_start = time.perf_counter()

    from training.scripts.driving_license_model.dl_stream_session import DlStreamSession
    from training.scripts.driving_license_model.dl_validator import get_yolo

    session = DlStreamSession(_DL_STREAM_W, _DL_STREAM_H, encode_params)
    session.start()
    invalid_ignore_remaining = 1 if _DL_IGNORE_FIRST_INVALID else 0
    try:
        model = get_yolo()
    except FileNotFoundError as e:
        logger.error("DL verify: %s", e)
        await websocket.close(code=1011)
        session.stop()
        return

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                while True:
                    try:
                        data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.0)
                    except asyncio.TimeoutError:
                        break
            except WebSocketDisconnect:
                break
            if not data:
                continue

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                frame = np.zeros((_DL_STREAM_H, _DL_STREAM_W, 3), dtype=np.uint8)
            h, w = frame.shape[:2]
            if (w, h) != (_DL_STREAM_W, _DL_STREAM_H):
                frame = cv2.resize(frame, (_DL_STREAM_W, _DL_STREAM_H), interpolation=cv2.INTER_LINEAR)

            t_inf0 = time.perf_counter()
            detections, jpeg_bytes = await loop.run_in_executor(
                _dl_stream_executor,
                session.fast_detect,
                frame,
                model,
                True,
            )
            inf_s = time.perf_counter() - t_inf0
            if inf_s > 3.0:
                logger.info("dl-verify: YOLO+annotate took %.1fs (detections=%s)", inf_s, len(detections))

            if jpeg_bytes is None:
                jpeg_bytes = _dl_placeholder_jpeg(encode_params)

            elapsed = time.perf_counter() - t_ws_start
            det_list: list[dict[str, Any]] = []
            for d in detections:
                det_list.append(
                    {
                        "bbox": d["bbox"],
                        "confidence": d["confidence"],
                        "class": d["class"],
                        "ocr_text": d.get("ocr_text", ""),
                        "ocr_lines": d.get("ocr_lines", []),
                        "validation_label": d.get("validation_label", "unknown"),
                        "validation_confidence": d.get("validation_confidence", 0.0),
                        "validation_reason": d.get("validation_reason", ""),
                        "dl_numbers": d.get("dl_numbers", []),
                        "validity_end": d.get("validity_end"),
                    }
                )

            payload: dict[str, Any] = {"detections": det_list}

            d0 = detections[0] if detections else None
            if d0:
                vl = str(d0.get("validation_label") or "unknown").lower()
                dl_nums = d0.get("dl_numbers") or []
                ocr_text = str(d0.get("ocr_text") or "")

                if vl == "invalid":
                    if invalid_ignore_remaining > 0:
                        invalid_ignore_remaining -= 1
                        logger.info(
                            "dl-verify: ignoring one invalid verdict (remaining free passes=%s pending_id=%s)",
                            invalid_ignore_remaining,
                            pending_id,
                        )
                        payload["registration"] = {
                            "status": "ignored",
                            "reason": "first_invalid_skipped",
                        }
                    else:
                        _pending_delete(pending_id)
                        payload["registration"] = {"status": "rejected", "reason": "invalid_dl"}
                        await websocket.send_bytes(jpeg_bytes)
                        await websocket.send_text(json.dumps(payload))
                        await websocket.close(code=1000)
                        return

                if (
                    vl == "valid"
                    and dl_nums
                    and ocr_text.strip()
                    and (
                        _DL_MIN_FINALIZE_SEC <= 0
                        or elapsed >= _DL_MIN_FINALIZE_SEC
                    )
                ):
                    dl_number_s = str(dl_nums[0]).strip()
                    ve = d0.get("validity_end")
                    ve_s = str(ve) if ve is not None else None
                    logger.info(
                        "dl-verify: valid licence — finalizing (name/age check), pending_id=%s",
                        pending_id,
                    )
                    fin = finalize_pending_registration_core(
                        pending_id,
                        verdict="valid",
                        dl_number=dl_number_s,
                        ocr_text=ocr_text,
                        validity_end=ve_s,
                    )
                    if fin.get("ok"):
                        payload["registration"] = {
                            "status": "completed",
                            "driver_id": fin.get("driver_id"),
                            "dl_number": fin.get("dl_number"),
                        }
                        await websocket.send_bytes(jpeg_bytes)
                        await websocket.send_text(json.dumps(payload))
                        await websocket.close(code=1000)
                        return
                    payload["registration"] = {"status": "error", "detail": fin.get("detail", "finalize failed")}
                    await websocket.send_bytes(jpeg_bytes)
                    await websocket.send_text(json.dumps(payload))
                    await websocket.close(code=1000)
                    return

                if (
                    _DL_MIN_FINALIZE_SEC > 0
                    and vl == "valid"
                    and dl_nums
                    and ocr_text.strip()
                    and elapsed < _DL_MIN_FINALIZE_SEC
                ):
                    payload["registration"] = {
                        "status": "waiting",
                        "reason": "min_session",
                        "min_session_sec": _DL_MIN_FINALIZE_SEC,
                        "elapsed_sec": round(elapsed, 1),
                        "seconds_remaining": round(max(0.0, _DL_MIN_FINALIZE_SEC - elapsed), 1),
                    }

            await websocket.send_bytes(jpeg_bytes)
            await websocket.send_text(json.dumps(payload))

    except WebSocketDisconnect:
        logger.info("DL verify WebSocket disconnected (pending_id=%s)", pending_id)
    except Exception as e:
        logger.error("DL verify WebSocket error: %s", e, exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        session.stop()
