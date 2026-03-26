"""
WebSocket client — run on your **local machine** while the API runs on the VM.

VM (`python -m app.api.main`): HTTP + WebSocket only.

This client:
  1. POST `/api/login/` — match face to DB (cosine similarity on embeddings).
  2. If not registered: terminal name/age → webcam (step 1 capture, step 2 progress bar on the
     captured frame while POST `/api/login/register` runs; server: MediaPipe → 3D embedding, DB save).
  3. New registration only: DL verification WebSocket (default: same API host ``/api/login/ws/dl-verify``;
     optional external ``driver_license_detection`` via ``--dl-ws``) → DB write on success (server-side
     finalize for the default stream; client POST ``finalize-dl`` only for legacy external service);
     any failure exits the session (no redirects).
  4. Frame-only confirmation, then **Do you want to Start Callibration?** — only **y** / **Y**.
  5. GET `/api/recalibrate`, then WebSocket `/api/stream`.

`--stream-only`: skip steps 1–4.

`--login-only`: one face login (`POST /api/login/`) then exit — for local API testing (no register/DL/stream).

`--register-only`: name/age + face capture + `POST /api/login/register`, then same DL websocket + `finalize-dl` as full flow, then exit (no calibration/stream).

Protocol (after stream starts):
    Client → Server: JPEG bytes per message
    Server → Client: JSON metrics, then composite JPEG (video + dashboard panel)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar
from urllib.parse import quote, urlparse, urljoin, parse_qs

# Reduce OpenCV Qt/GTK threading warnings when HighGUI uses Qt on Linux
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
# If Qt still spams the console: try `export OPENCV_HIGHGUI_BACKEND=GTK` before launch (build-dependent).
# Keep all OpenCV imshow/waitKey on one thread: use httpx.AsyncClient (no ThreadPoolExecutor / extra
# threads) for HTTP during embedding progress and DL finalize — avoids QObject::moveToThread on Qt builds.
# For `--register-only`, registration + DL use a single asyncio.run(...) so Qt is not split across
# two separate event loops.

import cv2

cv2.setNumThreads(0)
import httpx
import numpy as np
import websockets

from utils.api_limits import (
    HTTP_CLIENT_TIMEOUT_LOGIN_S,
    HTTP_CLIENT_TIMEOUT_RECALIBRATE_S,
    HTTP_CLIENT_TIMEOUT_REGISTER_S,
    HTTP_CLIENT_TIMEOUT_VERIFY_DL_S,
)
from utils.dl_license_vm_panel import (
    build_panel,
    extract_dl_number_for_finalize,
    select_primary_detection,
)
from utils.http_errors import httpx_response_detail
from utils.logger import get_logger

logger = get_logger(__name__)

# API paths relative to base URL (single source for client routes)
_PATH_API_LOGIN = "api/login/"
_PATH_API_REGISTER = "api/login/register"
_PATH_API_VERIFY_DL = "api/login/verify-dl"
_PATH_API_FINALIZE_DL = "api/login/finalize-dl"
_PATH_RECALIBRATE = "api/recalibrate"

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
SEND_JPEG_QUALITY = 75
RECV_TIMEOUT = 15.0

# Driving-licence phase: protocol matches driver_license_detection/vm_client.py (JPEG + JSON per frame).
# API ``/api/login/ws/dl-verify``: YOLO every frame, Qwen in a background thread (throttled). On ``valid``,
# the server runs name/age match + DB save and closes the socket (optional ``DL_VERIFY_MIN_SESSION_SEC`` delays that).
# Use a long recv timeout (default below) so slow OCR bursts do not desync the JPEG+JSON pairs.
_DL_VERIFY_WINDOW_TITLE = "Driving License Detection (VM)"
_DL_VERIFY_JPEG_QUALITY = int(os.environ.get("DL_VERIFY_JPEG_QUALITY", "50"))
# Default 300s: must exceed one server-side inference (Qwen + YOLO). Override with DL_VERIFY_RECV_TIMEOUT.
_DL_VERIFY_RECV_TIMEOUT_DEFAULT = 300.0

_encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), SEND_JPEG_QUALITY]


def _dl_verify_headless() -> bool:
    """No OpenCV windows during DL phase — avoids Qt ``QObject::moveToThread`` on many Linux builds."""
    return os.environ.get("DL_VERIFY_HEADLESS", "").strip().lower() in ("1", "true", "yes", "on")


_LOGIN_RETRIES = 1
_LOGIN_RETRY_DELAY_SEC = 0.8
_LOGIN_WINDOW_TITLE = "Login"
# ASCII only — Unicode em-dashes in window titles render as "?" on some OpenCV/Linux setups.
_REGISTER_CAPTURE_WINDOW = "Register - capture face"
_REGISTER_EMBED_WINDOW = "Register - face embedding"
_REGISTER_DL_WINDOW = "Register - driving licence"
_MESSAGE_OVERLAY_MS = 3500
_DASHBOARD_WINDOW = "Driver Safety System"

# Progress bar ramps toward this fraction while HTTP is in flight; completes at 1.0 when done.
_EMBEDDING_PROGRESS_RAMP_SEC = 5.0
_EMBEDDING_PROGRESS_CAP = 0.92
_EMBEDDING_DONE_HOLD_MS = 400
_CAPTURE_PREVIEW_MAX_BAD_READS = 150
_WARMUP_FRAMES = 30

T = TypeVar("T")


def _configure_capture(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _warmup_capture(cap: cv2.VideoCapture, n: int = _WARMUP_FRAMES) -> None:
    """Discard buffered/stale frames so the first visible frame is fresh."""
    for _ in range(n):
        cap.read()


def _letterbox_resize(bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Fit image into out_w x out_h with letterboxing (for embedding preview)."""
    if bgr is None or bgr.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    scale = min(out_w / w, out_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0 = (out_h - nh) // 2
    x0 = (out_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _draw_embedding_progress_bar(
    frame: np.ndarray,
    fraction: float,
    title: str = "Face embedding",
) -> None:
    """Bottom overlay: progress 0..1 and percent text (draws on `frame` in place)."""
    h, w = frame.shape[:2]
    fraction = max(0.0, min(1.0, fraction))
    bar_h = max(10, int(h * 0.028))
    margin_x = 20
    margin_bottom = 28
    y1 = h - margin_bottom
    y0 = y1 - bar_h
    x0 = margin_x
    x1 = w - margin_x
    bar_w = max(1, x1 - x0)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (35, 38, 42), -1)
    fill_w = int(bar_w * fraction)
    if fill_w > 0:
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y1), (0, 165, 90), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 185, 190), 1)
    pct = int(round(100 * fraction))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        f"{title}: {pct}%",
        (x0, y0 - 8),
        font,
        0.55,
        (0, 240, 255),
        2,
        cv2.LINE_AA,
    )


def _ws_url_with_driver(url: str, driver_id: Optional[str]) -> str:
    if not driver_id:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}driver_id={quote(driver_id, safe='')}"


def normalize_http_base_url(url: str) -> str:
    """Map ws(s):// API host to http(s):// so httpx can reach REST routes."""
    u = (url or "").strip()
    if u.startswith("ws://"):
        return "http://" + u[5:]
    if u.startswith("wss://"):
        return "https://" + u[6:]
    return u


def default_ws_url_from_base(base_url: str) -> str:
    base = normalize_http_base_url(base_url)
    p = urlparse(base.rstrip("/"))
    scheme = "wss" if p.scheme == "https" else "ws"
    netloc = p.netloc or "localhost:5000"
    return f"{scheme}://{netloc}/api/stream"


def server_dl_ws_url(base_url: str, pending_id: str) -> str:
    """
    Server-side DL pipeline (YOLO + Qwen on the API VM) — same host/port as ``--base-url``.

    Path ``/api/login/ws/dl-verify``; no separate DL service required.
    """
    base = normalize_http_base_url(base_url)
    p = urlparse(base.rstrip("/"))
    scheme = "wss" if p.scheme == "https" else "ws"
    netloc = p.netloc or "localhost:5000"
    return f"{scheme}://{netloc}/api/login/ws/dl-verify?pending_id={quote(pending_id)}&qwen=1"


def _is_connection_error(err: BaseException) -> bool:
    s = str(err).lower()
    return any(
        x in s
        for x in (
            "connection refused",
            "timed out",
            "timeout",
            "name or service not known",
            "failed to establish",
            "network is unreachable",
            "errno",
        )
    )


class ClientSessionBootstrap:
    """
    Local: login or register via API → confirmation frame → calibration prompt → stream.
    """

    def __init__(
        self,
        base_url: str,
        ws_url: str,
        source: str | int,
        driver_id: Optional[str] = None,
        dl_ws_url: str = "",
        *,
        dl_ws_explicit: bool = False,
    ) -> None:
        self.base_url = normalize_http_base_url(base_url).rstrip("/")
        self.ws_url = ws_url
        self.source = int(source) if isinstance(source, str) and str(source).isdigit() else source
        self.driver_id = driver_id
        self._logged_driver_id: Optional[str] = None
        self._pending_id: Optional[str] = None
        self._registered_name: Optional[str] = None
        self._registered_age: Optional[int] = None
        self.dl_ws_url = dl_ws_url
        # If False, DL WebSocket defaults to the API server ``/api/login/ws/dl-verify`` (recommended).
        # If True, ``dl_ws_url`` must point at an external driver_license_detection instance.
        self.dl_ws_explicit = dl_ws_explicit

    def _resolve_dl_ws_url(self) -> str:
        if self.dl_ws_explicit:
            return (self.dl_ws_url or "").strip()
        if self._pending_id:
            return server_dl_ws_url(self.base_url, self._pending_id)
        return ""

    def _api_url(self, path: str) -> str:
        return urljoin(self.base_url + "/", path.lstrip("/"))

    def _capture_single_frame(self) -> Tuple[Optional[bytes], Optional[np.ndarray]]:
        """One grab — used for login attempts (same as previous behaviour)."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", self.source)
            return None, None
        _configure_capture(cap)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            logger.error("Failed to read frame from source %s", self.source)
            return None, None
        ok, buf = cv2.imencode(".jpg", frame, _encode_params)
        if not ok:
            return None, None
        return buf.tobytes(), frame

    def _capture_preview_loop(
        self,
        cap: cv2.VideoCapture,
        *,
        window_title: str,
        primary_hint: str,
        secondary_hint: str,
        blank_message: str,
        on_max_bad_reads_log: str,
        primary_scale: float = 0.55,
        primary_color: Tuple[int, int, int] = (0, 255, 255),
        secondary_hint_y: int = 54,
    ) -> Tuple[Optional[bytes], Optional[np.ndarray]]:
        """
        Live preview; SPACE to capture, q to cancel. Retries reads instead of failing fast.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        bad_reads = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                bad_reads += 1
                if bad_reads >= _CAPTURE_PREVIEW_MAX_BAD_READS:
                    logger.error("%s", on_max_bad_reads_log)
                    return None, None
                blank = np.zeros((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), dtype=np.uint8)
                cv2.putText(blank, blank_message, (20, 40), font, 0.55, (0, 80, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_title, blank)
                cv2.waitKey(30)
                continue
            bad_reads = 0
            vis = frame.copy()
            cv2.putText(
                vis,
                primary_hint,
                (10, 28),
                font,
                primary_scale,
                primary_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                secondary_hint,
                (10, secondary_hint_y),
                font,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_title, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                ok, buf = cv2.imencode(".jpg", frame, _encode_params)
                if not ok:
                    return None, None
                return buf.tobytes(), frame
            if key == ord("q"):
                return None, None

    def _register_capture_preview_loop(
        self, cap: cv2.VideoCapture,
    ) -> Tuple[Optional[bytes], Optional[np.ndarray]]:
        """Face registration preview (step 1/2)."""
        return self._capture_preview_loop(
            cap,
            window_title=_REGISTER_CAPTURE_WINDOW,
            primary_hint="Step 1/2: Capture your face (camera on)",
            secondary_hint="SPACE: capture | q: cancel",
            blank_message="No camera frame - check device / permissions",
            on_max_bad_reads_log="Camera: too many failed reads during capture preview",
            secondary_hint_y=54,
        )

    def _driving_license_capture_loop(
        self, cap: cv2.VideoCapture,
    ) -> Tuple[Optional[bytes], Optional[np.ndarray]]:
        """Driving licence capture after registration."""
        return self._capture_preview_loop(
            cap,
            window_title=_REGISTER_DL_WINDOW,
            primary_hint="Show your Driving License",
            secondary_hint="SPACE: capture | q: cancel",
            blank_message="No camera frame - check device",
            on_max_bad_reads_log="Camera: too many failed reads during DL capture",
            primary_scale=0.6,
            primary_color=(0, 255, 200),
            secondary_hint_y=56,
        )

    async def _wait_for_http_with_camera_progress_async(
        self,
        cap: cv2.VideoCapture,
        http_coro: Callable[[], Awaitable[T]],
        window_title: str,
        status_line: str,
        progress_label: str = "Face embedding",
        snapshot_bgr: Optional[np.ndarray] = None,
    ) -> T:
        """
        While the server builds the embedding, keep the camera open and show a progress bar.

        HTTP uses ``httpx.AsyncClient`` on the **same asyncio thread** as ``imshow``/``waitKey``
        (no extra Python threads), which avoids Qt HighGUI ``QObject::moveToThread`` warnings.
        """
        t0 = time.time()
        disp_w, disp_h = CAPTURE_WIDTH, CAPTURE_HEIGHT
        font = cv2.FONT_HERSHEY_SIMPLEX

        def _compose_frame(*, hold_complete: bool) -> np.ndarray:
            if snapshot_bgr is not None:
                vis = _letterbox_resize(snapshot_bgr, disp_w, disp_h)
                cv2.putText(
                    vis,
                    "Captured frame (sent to server)",
                    (10, disp_h - 52),
                    font,
                    0.48,
                    (180, 200, 210),
                    1,
                    cv2.LINE_AA,
                )
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    frame = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "No camera frame - check device",
                        (20, 40),
                        font,
                        0.6,
                        (0, 80, 255),
                        2,
                        cv2.LINE_AA,
                    )
                vis = frame.copy()
            cv2.putText(
                vis,
                status_line,
                (10, 28),
                font,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if snapshot_bgr is not None:
                cv2.putText(
                    vis,
                    "Step 2/2: Face embedding (camera stays on)",
                    (10, 56),
                    font,
                    0.48,
                    (200, 220, 230),
                    1,
                    cv2.LINE_AA,
                )
            if hold_complete:
                bar_frac = 1.0
            else:
                elapsed = time.time() - t0
                bar_frac = min(
                    _EMBEDDING_PROGRESS_CAP,
                    elapsed / _EMBEDDING_PROGRESS_RAMP_SEC * _EMBEDDING_PROGRESS_CAP,
                )
            _draw_embedding_progress_bar(vis, bar_frac, progress_label)
            return vis

        try:
            task = asyncio.create_task(http_coro())
            while not task.done():
                if snapshot_bgr is not None:
                    cap.read()
                vis = _compose_frame(hold_complete=False)
                cv2.imshow(window_title, vis)
                cv2.waitKey(1)
                await asyncio.sleep(0)

            result = await task

            end = time.time() + _EMBEDDING_DONE_HOLD_MS / 1000.0
            while time.time() < end:
                if snapshot_bgr is not None:
                    cap.read()
                vis = _compose_frame(hold_complete=True)
                cv2.imshow(window_title, vis)
                cv2.waitKey(1)
                await asyncio.sleep(0)

            return result
        finally:
            try:
                cv2.destroyWindow(window_title)
            except Exception:
                pass

    def _wait_for_http_with_camera_progress(
        self,
        cap: cv2.VideoCapture,
        http_coro: Callable[[], Awaitable[T]],
        window_title: str,
        status_line: str,
        progress_label: str = "Face embedding",
        snapshot_bgr: Optional[np.ndarray] = None,
    ) -> T:
        """Sync wrapper (runs asyncio event loop) for login/register embedding progress UI."""
        return asyncio.run(
            self._wait_for_http_with_camera_progress_async(
                cap,
                http_coro,
                window_title,
                status_line,
                progress_label=progress_label,
                snapshot_bgr=snapshot_bgr,
            )
        )

    def _show_timed_message_frame(
        self, frame: np.ndarray, message: str, window_title: str,
    ) -> None:
        vis = frame.copy()
        h, w = vis.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.7, min(min(w, h) / 900.0, 1.2))
        thickness = max(2, int(2 * scale))
        (tw, th), _ = cv2.getTextSize(message, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(vis, message, (x, y), font, scale, (40, 220, 40), thickness, cv2.LINE_AA)
        cv2.imshow(window_title, vis)
        cv2.waitKey(_MESSAGE_OVERLAY_MS)
        try:
            cv2.destroyWindow(window_title)
        except Exception:
            pass

    async def _login_http_async(self, image_bytes: bytes) -> Dict[str, Any]:
        login_path = self._api_url(_PATH_API_LOGIN)
        files = {"image": ("login_frame.jpg", image_bytes, "image/jpeg")}
        data: Dict[str, str] = {}
        if self.driver_id:
            data["driver_id"] = self.driver_id

        logger.info(
            "POST %s (image=%d bytes, driver_id=%s)",
            login_path,
            len(image_bytes),
            self.driver_id,
        )
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_LOGIN_S, follow_redirects=False) as client:
            response = await client.post(login_path, files=files, data=data or None)

        if response.status_code == 200:
            body = response.json()
            logger.info("Login API success: driver_id=%s", body.get("driver_id"))
            return body

        err = httpx_response_detail(response)
        logger.warning("Login API failed: status=%s detail=%s", response.status_code, err)
        raise RuntimeError(f"Login failed ({response.status_code}): {err}")

    async def _register_http_async(self, name: str, age: Optional[int], image_bytes: bytes) -> Dict[str, Any]:
        """POST /api/login/register — MediaPipe + Open3D embedding; pending_id only until finalize-dl (no DB row yet)."""
        url = self._api_url(_PATH_API_REGISTER)
        payload: Dict[str, Any] = {
            "name": name,
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        }
        if age is not None:
            payload["age"] = age

        logger.info("POST %s (name=%s, age=%s)", url, name, age)
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_REGISTER_S, follow_redirects=False) as client:
            response = await client.post(url, json=payload)

        if response.status_code == 200:
            body = response.json()
            logger.info("Registration success: driver_id=%s", body.get("driver_id"))
            return body

        err = httpx_response_detail(response)
        logger.warning("Register API failed: status=%s detail=%s", response.status_code, err)
        raise RuntimeError(f"Register failed ({response.status_code}): {err}")

    def _verify_dl_http(self, image_bytes: bytes) -> Dict[str, Any]:
        """POST /api/login/verify-dl — fallback single-shot strict verification."""
        url = self._api_url(_PATH_API_VERIFY_DL)
        files = {"image": ("dl.jpg", image_bytes, "image/jpeg")}
        data = {"pending_id": self._pending_id or ""}
        logger.info("POST %s (pending_id=%s, image=%d bytes)", url, self._pending_id, len(image_bytes))
        with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT_VERIFY_DL_S, follow_redirects=False) as client:
            response = client.post(url, files=files, data=data)

        if response.status_code == 200:
            body = response.json()
            logger.info("verify-dl success: dl_number=%s", body.get("dl_number"))
            return body

        err = httpx_response_detail(response)
        logger.warning("verify-dl failed: status=%s detail=%s", response.status_code, err)
        raise RuntimeError(f"{response.status_code}: {err}")

    async def _finalize_dl_http_async(
        self,
        *,
        verdict: str,
        dl_number: str,
        validity_end: Optional[str],
        ocr_text: str,
    ) -> Dict[str, Any]:
        """POST /api/login/finalize-dl — uses DL websocket results; performs the *first* MongoDB write."""
        url = self._api_url(_PATH_API_FINALIZE_DL)
        data: Dict[str, Any] = {
            "pending_id": self._pending_id or "",
            "verdict": verdict,
            "dl_number": dl_number,
            "ocr_text": ocr_text or "",
        }
        if validity_end:
            data["validity_end"] = validity_end

        logger.info("POST %s (pending_id=%s, dl_number=%s)", url, self._pending_id, dl_number)
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_VERIFY_DL_S, follow_redirects=False) as client:
            response = await client.post(url, data=data)

        if response.status_code == 200:
            body = response.json()
            logger.info("finalize-dl success: driver_id=%s dl_number=%s", body.get("driver_id"), body.get("dl_number"))
            return body

        err = httpx_response_detail(response)
        logger.warning("finalize-dl failed: status=%s detail=%s", response.status_code, err)
        raise RuntimeError(f"{response.status_code}: {err}")

    def _trigger_recalibrate(self) -> None:
        rec_url = self._api_url(_PATH_RECALIBRATE)
        logger.info("GET %s", rec_url)
        with httpx.Client(timeout=HTTP_CLIENT_TIMEOUT_RECALIBRATE_S, follow_redirects=False) as client:
            response = client.get(rec_url)
        if response.status_code != 200:
            logger.error("Recalibrate failed: status=%s body=%s", response.status_code, response.text)
            raise RuntimeError(f"Recalibrate failed: {response.status_code}")
        logger.info("Recalibrate OK: %s", response.text)

    def _read_name_age_terminal(self) -> Tuple[str, Optional[int]]:
        name = input("Enter your full name: ").strip()
        while not name:
            name = input("Name is required. Enter your full name: ").strip()

        age_raw = input("Age (optional, press Enter to skip): ").strip()
        age: Optional[int] = None
        if age_raw:
            try:
                age = int(age_raw)
                if not (1 <= age <= 120):
                    logger.warning("Age out of range 1–120, ignored")
                    age = None
            except ValueError:
                logger.warning("Invalid age, ignored")
                age = None
        return name, age

    def _phase_login(self) -> bool:
        """Return True if logged in."""
        logger.info(
            "Checking registration (face embedding in database, cosine similarity).",
        )
        logger.info("Capturing face — look at the camera.")
        last_err: Optional[Exception] = None
        saw_connection_error = False

        for attempt in range(1, _LOGIN_RETRIES + 1):
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logger.error("Login attempt %d: cannot open video source %s", attempt, self.source)
                last_err = RuntimeError("Could not open camera")
                continue
            _configure_capture(cap)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                logger.error("Login capture attempt %d failed (no frame)", attempt)
                last_err = RuntimeError("Could not capture image")
                continue
            bgr = frame.copy()
            ok, buf = cv2.imencode(".jpg", frame, _encode_params)
            if not ok:
                cap.release()
                logger.error("Login capture attempt %d failed (encode)", attempt)
                last_err = RuntimeError("Could not encode image")
                continue
            image_bytes = buf.tobytes()
            try:
                result = self._wait_for_http_with_camera_progress(
                    cap,
                    lambda: self._login_http_async(image_bytes),
                    _LOGIN_WINDOW_TITLE,
                    "Camera on - matching face (server builds embedding)",
                    progress_label="Login",
                    snapshot_bgr=bgr,
                )
                self._logged_driver_id = result.get("driver_id")
                logger.info("Face matched: driver_id=%s", self._logged_driver_id)
                logger.info("Login Completed (match)")
                try:
                    self._show_timed_message_frame(bgr, "Login Completed", _LOGIN_WINDOW_TITLE)
                except Exception as exc:
                    logger.warning("Login frame preview unavailable: %s", exc)
                return True
            except httpx.RequestError as e:
                last_err = e
                saw_connection_error = True
                logger.warning("Login attempt %d/%d (HTTP transport): %s", attempt, _LOGIN_RETRIES, e)
                if attempt < _LOGIN_RETRIES:
                    logger.info("Cannot reach server — retrying...")
                    time.sleep(_LOGIN_RETRY_DELAY_SEC)
            except RuntimeError as e:
                last_err = e
                if _is_connection_error(e):
                    saw_connection_error = True
                logger.warning("Login attempt %d/%d: %s", attempt, _LOGIN_RETRIES, e)
                if attempt < _LOGIN_RETRIES:
                    logger.info(
                        "No match or not registered — adjust position/lighting. Retrying...",
                    )
                    time.sleep(_LOGIN_RETRY_DELAY_SEC)
            finally:
                cap.release()

        if saw_connection_error or (last_err and _is_connection_error(last_err)):
            logger.error(
                "Cannot reach the API server. Check --base-url and network.",
            )
            if last_err:
                logger.error("Login aborted (connection): %s", last_err)
            sys.exit(1)

        logger.info("Not registered — you can register as a new driver.")
        if last_err:
            logger.info("Login attempts exhausted: %s", last_err)
        return False

    def _phase_register(self) -> bool:
        """Return True if registration succeeded and pending_id set."""
        return asyncio.run(self._phase_register_async())

    async def _phase_register_async(self) -> bool:
        """Return True if registration succeeded and pending_id set (async; no nested asyncio.run)."""
        name, age = self._read_name_age_terminal()
        logger.info(
            "Opening webcam - step 1/2: capture. Step 2/2: embedding on server (progress bar). "
            "Press SPACE to capture, q to cancel.",
        )
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Cannot open video source for registration: %s", self.source)
            logger.error("Could not open camera. Check camera index and permissions.")
            return False
        _configure_capture(cap)
        _warmup_capture(cap)
        try:
            cv2.namedWindow(_REGISTER_CAPTURE_WINDOW, cv2.WINDOW_AUTOSIZE)
        except Exception:
            pass

        jpeg_bytes, bgr = self._register_capture_preview_loop(cap)
        if not jpeg_bytes or bgr is None:
            logger.warning("Registration cancelled or capture failed.")
            cap.release()
            try:
                cv2.destroyWindow(_REGISTER_CAPTURE_WINDOW)
            except Exception:
                pass
            return False

        try:
            cv2.destroyWindow(_REGISTER_CAPTURE_WINDOW)
        except Exception:
            pass

        logger.info(
            "Sending snapshot to server - embedding in progress (watch the progress bar)...",
        )
        try:
            cv2.namedWindow(_REGISTER_EMBED_WINDOW, cv2.WINDOW_AUTOSIZE)
        except Exception:
            pass

        try:
            out = await self._wait_for_http_with_camera_progress_async(
                cap,
                lambda: self._register_http_async(name, age, jpeg_bytes),
                _REGISTER_EMBED_WINDOW,
                "Camera on - creating face embedding (server)",
                progress_label="Embedding",
                snapshot_bgr=bgr,
            )
        except httpx.RequestError as e:
            logger.error("Registration failed (HTTP): %s", e)
            return False
        except RuntimeError as e:
            logger.error("Registration failed: %s", e)
            return False
        finally:
            cap.release()

        self._pending_id = out.get("pending_id")
        self._registered_name = name
        self._registered_age = age
        logger.info("Registration pending. pending_id=%s", self._pending_id)
        try:
            self._show_timed_message_frame(bgr, "Registration Completed", _LOGIN_WINDOW_TITLE)
        except Exception as exc:
            logger.warning("Registration confirmation frame: %s", exc)
        return True

    def _phase_verify_driving_license(self) -> bool:
        """After new registration: run full DL websocket pipeline until valid."""
        return asyncio.run(self._phase_verify_driving_license_async())

    async def _phase_verify_driving_license_async(self) -> bool:
        """DL websocket + finalize-dl; use await (no nested asyncio.run)."""
        if not self._pending_id:
            return False

        logger.info("Driving licence verification — streaming websocket pipeline...")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            ok, last_bgr = await self._dl_stream_and_finalize()
        except RuntimeError as e:
            logger.error("DL websocket verification failed to start: %s", e)
            return False

        if ok and last_bgr is not None:
            try:
                self._show_timed_message_frame(last_bgr, "DL Verified", _LOGIN_WINDOW_TITLE)
            except Exception as exc:
                logger.warning("DL verified frame: %s", exc)
        return ok

    async def _register_only_async(self) -> bool:
        """Single event loop for register + DL (reduces Qt HighGUI thread warnings)."""
        if not await self._phase_register_async():
            return False
        return await self._phase_verify_driving_license_async()

    async def _dl_stream_and_finalize(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Stream JPEG frames to the DL WebSocket; receive annotated frame + JSON.

        **Default:** same host as ``--base-url`` → ``/api/login/ws/dl-verify`` (YOLO+Qwen+finalize on server).

        **Legacy:** set ``--dl-ws`` or ``DL_DETECTION_WS_URL`` for a separate driver_license_detection service;
        then the client POSTs ``/finalize-dl`` when the licence is valid.
        """
        ws_url = self._resolve_dl_ws_url()
        if not ws_url:
            logger.error("DL websocket URL not configured (need pending_id for server stream or --dl-ws).")
            return False, None

        parsed = urlparse(ws_url)
        qs = parse_qs(parsed.query)
        qwen_enabled = qs.get("qwen", ["0"])[0] == "1"

        if not qwen_enabled:
            logger.error("DL websocket URL must include `?qwen=1` (same as driver_license_detection).")
            return False, None

        recv_to = float(
            os.environ.get("DL_VERIFY_RECV_TIMEOUT", str(_DL_VERIFY_RECV_TIMEOUT_DEFAULT)),
        )
        target_fps = float(os.environ.get("DL_VERIFY_TARGET_FPS", "2.0"))
        min_interval = 1.0 / target_fps if target_fps > 0 else 0.0

        headless = _dl_verify_headless()
        logger.info(
            "Connecting to DL websocket: %s (qwen_enabled=%s, external_service=%s, headless=%s, recv_timeout=%.1fs)",
            ws_url,
            qwen_enabled,
            self.dl_ws_explicit,
            headless,
            recv_to,
        )

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Cannot open camera for DL verification")
            return False, None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        _configure_capture(cap)

        last_bgr: Optional[np.ndarray] = None
        last_info: dict[str, Any] = {"detections": []}
        fps_t0 = time.time()
        fps_count = 0
        fps = 0.0

        encode_params_dl = [int(cv2.IMWRITE_JPEG_QUALITY), _DL_VERIFY_JPEG_QUALITY]

        _dl_connect_timeout = float(os.environ.get("DL_VERIFY_CONNECT_TIMEOUT", "30"))
        try:
            async with websockets.connect(
                ws_url,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
                open_timeout=_dl_connect_timeout,
            ) as ws:
                last_send_t = 0.0

                while True:
                    now = time.time()
                    if min_interval and (now - last_send_t) < min_interval:
                        await asyncio.sleep(min_interval - (now - last_send_t))
                    last_send_t = time.time()

                    # Read on the asyncio thread (same thread as imshow/waitKey). Executor + Qt
                    # HighGUI causes QObject::moveToThread on many Linux opencv builds.
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to read webcam frame during DL verification")
                        return False, last_bgr
                    if frame is None:
                        continue

                    frame = cv2.resize(frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT))
                    last_bgr = frame.copy()

                    ok, buffer = cv2.imencode(".jpg", frame, encode_params_dl)
                    if not ok:
                        continue

                    await ws.send(buffer.tobytes())

                    try:
                        msg1 = await asyncio.wait_for(ws.recv(), timeout=recv_to)
                        if qwen_enabled:
                            msg2 = await asyncio.wait_for(ws.recv(), timeout=recv_to)
                        else:
                            msg2 = None
                    except asyncio.TimeoutError:
                        continue

                    annotated_img: Any = None
                    json_str: str | None = None

                    if isinstance(msg1, (bytes, bytearray)):
                        annotated_bytes = bytes(msg1)
                        annotated_img = cv2.imdecode(
                            np.frombuffer(annotated_bytes, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        if qwen_enabled and isinstance(msg2, str):
                            json_str = msg2
                    elif qwen_enabled and isinstance(msg1, str):
                        json_str = msg1

                    if annotated_img is None:
                        annotated_img = frame

                    session_reg: dict[str, Any] = {}
                    if json_str:
                        try:
                            parsed_json = json.loads(json_str)
                            session_reg = parsed_json.get("registration") or {}
                            dets = parsed_json.get("detections") or []
                            if not dets:
                                last_info = {"detections": []}
                            else:
                                last_info = parsed_json
                        except json.JSONDecodeError as e:
                            logger.warning("Failed to parse DL server JSON: %s", e)

                    dets = last_info.get("detections") or []
                    d0 = select_primary_detection(dets if isinstance(dets, list) else [])
                    verdict = str(d0.get("validation_label") or "unknown").lower()

                    fps_count += 1
                    now = time.time()
                    elapsed = now - fps_t0
                    if elapsed >= 1.0:
                        fps = fps_count / elapsed
                        fps_count = 0
                        fps_t0 = now

                    if not headless:
                        panel = build_panel(last_info, fps, annotated_img.shape[0])
                        combined = np.hstack([annotated_img, panel])
                        cv2.imshow(_DL_VERIFY_WINDOW_TITLE, combined)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("x"):
                            return False, last_bgr
                    else:
                        if fps_count % max(1, int(target_fps * 2)) == 0:
                            logger.info(
                                "DL frame: verdict=%s registration=%s",
                                verdict,
                                session_reg.get("status") if session_reg else "-",
                            )

                    # Server-side stream finalizes in-process; JSON may include ``registration``.
                    rs = session_reg.get("status")
                    if rs == "completed":
                        self._logged_driver_id = session_reg.get("driver_id")
                        self._pending_id = None
                        if not self.dl_ws_explicit:
                            logger.info(
                                "DL verification complete — server saved driver (name/age checked); "
                                "driver_id=%s",
                                self._logged_driver_id,
                            )
                        return True, last_bgr
                    if rs == "rejected":
                        if not headless:
                            try:
                                self._show_timed_message_frame(last_bgr, "DL Invalid", _LOGIN_WINDOW_TITLE)
                            except Exception:
                                pass
                        else:
                            logger.warning("DL rejected (invalid licence)")
                        return False, last_bgr
                    if rs == "error":
                        detail = str(session_reg.get("detail") or "DL Verification Failed")
                        logger.error("Server finalize failed: %s", detail)
                        if not headless:
                            try:
                                self._show_timed_message_frame(
                                    last_bgr,
                                    "Mismatched Name" if "Mismatched Name" in detail else (
                                        "Mismatched Age" if "Mismatched Age" in detail else "DL Verification Failed"
                                    ),
                                    _LOGIN_WINDOW_TITLE,
                                )
                            except Exception:
                                pass
                        return False, last_bgr

                    # External driver_license_detection only: client must POST /finalize-dl.
                    if self.dl_ws_explicit and verdict == "invalid":
                        if not headless:
                            try:
                                self._show_timed_message_frame(last_bgr, "DL Invalid", _LOGIN_WINDOW_TITLE)
                            except Exception:
                                pass
                        try:
                            await self._finalize_dl_http_async(
                                verdict="invalid",
                                dl_number="",
                                validity_end=None,
                                ocr_text="",
                            )
                        except RuntimeError:
                            pass
                        return False, last_bgr

                    if self.dl_ws_explicit and verdict == "valid":
                        ocr_text = str(d0.get("ocr_text") or "")
                        dl_number = extract_dl_number_for_finalize(d0)
                        validity_end = d0.get("validity_end")
                        if dl_number and ocr_text:
                            try:
                                out = await self._finalize_dl_http_async(
                                    verdict="valid",
                                    dl_number=dl_number,
                                    validity_end=str(validity_end) if validity_end else None,
                                    ocr_text=ocr_text,
                                )
                                self._logged_driver_id = out.get("driver_id")
                                self._pending_id = None
                                return True, last_bgr
                            except httpx.RequestError as e:
                                logger.error("finalize-dl HTTP error: %s", e)
                                return False, last_bgr
                            except RuntimeError as e:
                                err = str(e)
                                logger.error("finalize-dl failed: %s", err)
                                if not headless:
                                    if "Mismatched Name" in err:
                                        msg = "Mismatched Name"
                                    elif "Mismatched Age" in err:
                                        msg = "Mismatched Age"
                                    else:
                                        msg = "DL Verification Failed"
                                    try:
                                        self._show_timed_message_frame(last_bgr, msg, _LOGIN_WINDOW_TITLE)
                                    except Exception:
                                        pass
                                return False, last_bgr

        except (OSError, TimeoutError) as e:
            hint = (
                " For the default server-side DL stream, ensure the API (same host as --base-url) is reachable."
                if not self.dl_ws_explicit
                else " Ensure the external DL service is running (e.g. open firewall for DL_DETECTION_PORT, often 5001)."
            )
            logger.error(
                "Cannot connect to DL websocket %s: %r (errno=%s).%s",
                ws_url,
                e,
                getattr(e, "errno", None),
                hint,
            )
            return False, last_bgr
        except Exception as e:
            logger.error("DL verification websocket error: %s", e, exc_info=True)
            return False, last_bgr
        finally:
            cap.release()
            try:
                cv2.destroyWindow(_DL_VERIFY_WINDOW_TITLE)
            except Exception:
                pass

        return False, last_bgr

    def _phase_calibration_and_stream(self) -> None:
        answer = input("Do you want to Start Callibration? [y/N]: ").strip()
        if answer != "y" and answer != "Y":
            logger.info("Calibration not started; stream not opened.")
            logger.info("Exiting.")
            return

        try:
            self._trigger_recalibrate()
        except RuntimeError as e:
            logger.error("Calibration request failed: %s", e)
            sys.exit(1)

        cv2.destroyAllWindows()
        ws_driver = self._logged_driver_id or self.driver_id
        logger.info("Opening WebSocket stream (frame + dashboard from server)")
        asyncio.run(stream(self.ws_url, self.source, driver_id=ws_driver))
        logger.info("Session ended")

    def run(self, *, login_only: bool = False, register_only: bool = False) -> None:
        """
        Full bootstrap: login → optional register + DL → calibration → stream.
        If `login_only` is True, run a single login attempt and exit (for API/backend testing;
        frontend owns register/stream navigation).
        If `register_only` is True, run registration then driving-licence verification (websocket + finalize-dl), then exit — no calibration/stream.
        """
        if login_only:
            ok = self._phase_login()
            sys.exit(0 if ok else 1)

        if register_only:
            ok = asyncio.run(self._register_only_async())
            if not ok:
                if not self._pending_id:
                    logger.error("Registration failed or was cancelled.")
                else:
                    logger.error("Driving licence verification failed — exiting session.")
                sys.exit(1)
            logger.info(
                "Register-only finished: driver saved (driver_id=%s). Exiting.",
                self._logged_driver_id,
            )
            sys.exit(0)

        registered_this_session = False
        if not self._phase_login():
            if not self._phase_register():
                sys.exit(1)
            registered_this_session = True
        if registered_this_session:
            if not self._phase_verify_driving_license():
                logger.error("Driving licence verification failed — exiting session.")
                sys.exit(1)
        self._phase_calibration_and_stream()


async def stream(url: str, source, driver_id: Optional[str] = None):
    ws_connect_url = _ws_url_with_driver(url, driver_id)
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(source)
    _configure_capture(cap)

    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        sys.exit(1)

    logger.info("Connecting to %s (source=%s, driver_id=%s)", ws_connect_url, source, driver_id)

    try:
        ws = await asyncio.wait_for(
            websockets.connect(ws_connect_url, max_size=None), timeout=10.0
        )
    except asyncio.TimeoutError:
        logger.error("Connection timed out — is the server running at %s?", url)
        sys.exit(1)
    except OSError as e:
        logger.error("Cannot connect to %s: %s", url, e)
        sys.exit(1)
    except Exception as e:
        logger.error("WebSocket connection failed: %s", e, exc_info=True)
        sys.exit(1)

    try:
        logger.info("Connected — streaming frames...")
        fps_time = time.time()
        fps_count = 0
        fps_display = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode(".jpg", frame, _encode_params)
            try:
                await ws.send(buffer.tobytes())
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("Server closed connection during send: %s", e.reason or e.code)
                break

            try:
                metrics_raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error("No metrics response (timeout=%.1fs)", RECV_TIMEOUT)
                break
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("Server closed connection waiting for metrics: %s", e.reason or e.code)
                break
            try:
                json.loads(metrics_raw)
            except json.JSONDecodeError:
                pass

            try:
                data = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error("No frame response (timeout=%.1fs)", RECV_TIMEOUT)
                break
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("Server closed connection waiting for frame: %s", e.reason or e.code)
                break

            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                fps_count += 1
                now = time.time()
                elapsed = now - fps_time
                if elapsed >= 1.0:
                    fps_display = fps_count / elapsed
                    fps_count = 0
                    fps_time = now

                cv2.putText(
                    img,
                    f"{fps_display:.0f} FPS",
                    (img.shape[1] - 90, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (120, 125, 130),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(_DASHBOARD_WINDOW, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("x"):
                break
            if key == ord("r"):
                logger.info("Recalibration requested by user (use GET /api/recalibrate on server)")
    finally:
        try:
            await ws.close()
        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Client shutdown complete")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Local client: login or register → calibration → WebSocket (VM API).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:5000",
        help="HTTP base URL of the VM API (login, register, /api/recalibrate)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="WebSocket URL (default: derived from --base-url)",
    )
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument(
        "--driver-id",
        default=None,
        help="Optional: restrict login to this driver_id",
    )
    parser.add_argument(
        "--stream-only",
        action="store_true",
        help="Skip login/register/calibration; connect WebSocket immediately",
    )
    parser.add_argument(
        "--login-only",
        action="store_true",
        help="Only run face login (POST /api/login/) once, then exit — no register, DL, or stream",
    )
    parser.add_argument(
        "--register-only",
        action="store_true",
        help=(
            "Registration (name/age + face) + POST /api/login/register, then DL websocket verification "
            "and finalize-dl (same as full flow); then exit — no calibration or stream"
        ),
    )
    parser.add_argument(
        "--dl-ws",
        default=None,
        help=(
            "Optional: external driver_license_detection WebSocket (must include ?qwen=1). "
            "If omitted, DL runs on the API server at /api/login/ws/dl-verify (same host as --base-url)."
        ),
    )
    parser.add_argument(
        "--dl-headless",
        action="store_true",
        help=(
            "During DL verification: do not open OpenCV windows (avoids Qt QObject::moveToThread on some Linux setups); "
            "log status to the terminal instead. Same as DL_VERIFY_HEADLESS=1."
        ),
    )
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    source = int(args.source) if str(args.source).isdigit() else args.source
    base_url = normalize_http_base_url(args.base_url)
    ws_url = args.url or default_ws_url_from_base(base_url)
    dl_env = (os.environ.get("DL_DETECTION_WS_URL") or "").strip()
    dl_arg = (args.dl_ws or "").strip()
    dl_ws_explicit = bool(dl_arg) or bool(dl_env)
    dl_ws_url = dl_arg or dl_env

    _exclusive = int(args.stream_only) + int(args.login_only) + int(args.register_only)
    if _exclusive > 1:
        raise SystemExit("Use only one of: --stream-only, --login-only, --register-only")

    if args.stream_only:
        asyncio.run(stream(ws_url, source, driver_id=args.driver_id))
        return

    if getattr(args, "dl_headless", False):
        os.environ["DL_VERIFY_HEADLESS"] = "1"

    bootstrap = ClientSessionBootstrap(
        base_url=base_url,
        ws_url=ws_url,
        source=source,
        driver_id=args.driver_id,
        dl_ws_url=dl_ws_url,
        dl_ws_explicit=dl_ws_explicit,
    )
    bootstrap.run(login_only=args.login_only, register_only=args.register_only)


if __name__ == "__main__":
    main()
