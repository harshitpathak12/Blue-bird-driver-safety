"""
Server-side overlay and dashboard panel for the Driver Monitoring System.

Composes: face annotations on the video + a dashboard panel to the right.
The server sends the combined image so any client can display it directly.
"""

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

PANEL_WIDTH = 340
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Face annotation colours ───────────────────────────────────────────
_BBOX_COLOR = (0, 230, 200)
_LANDMARK_COLOR = (220, 230, 240)
_EYE_RING_COLOR = (200, 180, 100)

_LEFT_EYE_IDS = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_IDS = [362, 385, 387, 263, 373, 380]
_KEY_LANDMARK_IDS = [
    1, 2, 61, 291, 0, 17,
    33, 133, 159, 145,
    362, 263, 386, 374,
    10, 152, 234, 454,
]

# ── Dashboard palette (BGR) ───────────────────────────────────────────
_BG = (25, 25, 25)
_WHITE = (240, 240, 240)
_LIGHT = (190, 195, 200)
_GRAY = (120, 125, 130)
_DARK_GRAY = (60, 60, 60)
_GREEN = (0, 200, 100)
_YELLOW = (0, 210, 255)
_RED = (0, 70, 255)
_CYAN = (230, 190, 0)
_BAR_BG = (50, 50, 50)
_ALERT_BG = (0, 0, 110)


# ── Drawing primitives ────────────────────────────────────────────────
def _t(img, text, x, y, scale=0.44, color=_WHITE, thick=1):
    cv2.putText(img, str(text), (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, pct, fg):
    cv2.rectangle(img, (x, y), (x + w, y + h), _BAR_BG, -1)
    fill = int(w * max(0.0, min(1.0, pct)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), fg, -1)


def _div(img, y, w):
    cv2.line(img, (12, y), (w - 12, y), _DARK_GRAY, 1)


def _level_color(pct):
    if pct > 60:
        return _RED
    if pct > 30:
        return _YELLOW
    return _GREEN


def _vote_color(v):
    return _RED if v == "distraction" else _GREEN


def _state_appearance(state):
    return {
        "NORMAL": ("All Good", _GREEN),
        "FATIGUE": ("Fatigue Detected", _YELLOW),
        "DISTRACTION": ("Distracted!", _RED),
        "SLEEP": ("Falling Asleep!", _RED),
        "WAITING": ("Waiting...", _GRAY),
        "NO_FACE": ("No Face", _GRAY),
        "CALIBRATING": ("Calibrating...", _CYAN),
    }.get(state, (state, _WHITE))


def _alert_message(alert_type):
    return {
        "fatigue": "TAKE A BREAK!",
        "distraction": "LOOK AT THE ROAD!",
        "sleep": "WAKE UP!",
    }.get((alert_type or "").lower(), "WARNING!")


# ── Face annotation (drawn on the video frame) ───────────────────────
def _draw_face_overlays(frame, landmarks, img_w, img_h):
    if landmarks is None or len(landmarks) < 5:
        return

    pts = [(int(lm[0] * img_w), int(lm[1] * img_h)) for lm in landmarks]
    if not pts:
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 20
    x1, x2 = max(0, min(xs) - pad), min(img_w, max(xs) + pad)
    y1, y2 = max(0, min(ys) - pad), min(img_h, max(ys) + pad)

    for i in range(3):
        cv2.rectangle(frame, (x1 - i, y1 - i), (x2 + i, y2 + i), _BBOX_COLOR, 1)

    for i in _KEY_LANDMARK_IDS:
        if i < len(pts):
            cv2.circle(frame, pts[i], 2, _LANDMARK_COLOR, -1)

    for eye_ids in (_LEFT_EYE_IDS, _RIGHT_EYE_IDS):
        if max(eye_ids) >= len(landmarks):
            continue
        ex = int(np.mean([landmarks[i][0] * img_w for i in eye_ids]))
        ey = int(np.mean([landmarks[i][1] * img_h for i in eye_ids]))
        cv2.circle(frame, (ex, ey), 12, _EYE_RING_COLOR, 2)
        cv2.line(frame, (ex - 5, ey), (ex + 5, ey), _LANDMARK_COLOR, 1)
        cv2.line(frame, (ex, ey - 5), (ex, ey + 5), _LANDMARK_COLOR, 1)


def draw_face_overlay(frame, landmarks, img_w, img_h):
    """Public alias for backward compatibility."""
    _draw_face_overlays(frame, landmarks, img_w, img_h)


# ── Dashboard panel (separate image, hstacked by the renderer) ───────
def _build_panel(metrics: dict, frame_h: int) -> np.ndarray:
    w = PANEL_WIDTH
    px = 16
    bar_w = w - 2 * px - 60
    line_gap = 18
    section_gap = 20
    bar_h = 10
    panel = np.full((frame_h, w, 3), _BG, dtype=np.uint8)
    y = 28

    # Title
    _t(panel, "Driver Safety Monitor", px, y, 0.58, _CYAN, 2)
    y += 10
    _div(panel, y, w)
    y += section_gap

    # Driver identity
    identity = metrics.get("driver_identity")
    if identity and identity not in ("\u2014", "Unknown"):
        _t(panel, f"Driver: {str(identity)[:24]}", px, y, 0.42, _LIGHT)
        y += line_gap

    # Verified driving licence number (from DB after registration)
    _t(panel, "DL Number", px, y, 0.42, _GRAY)
    y += line_gap
    dln = metrics.get("dl_number")
    if dln and str(dln).strip() and str(dln) != "\u2014":
        _t(panel, str(dln)[:26], px, y, 0.42, _GREEN)
    else:
        _t(panel, "—", px, y, 0.42, _LIGHT)
    y += line_gap

    # Status / calibration
    attn = (metrics.get("attention_state") or "attentive").lower()
    if attn == "calibrating":
        remaining = metrics.get("calibration_remaining_sec", 0)
        _t(panel, "CALIBRATING", px, y, 0.55, _CYAN, 2)
        y += line_gap
        _t(panel, f"Look forward  ({remaining:.0f}s left)", px, y, 0.40, _LIGHT)
        y += line_gap
        _div(panel, y, w)
        y += 14
    else:
        state = metrics.get("driver_state", "waiting").upper()
        label, color = _state_appearance(state)
        _t(panel, label, px, y, 0.58, color, 2)
        y += line_gap
        _div(panel, y, w)
        y += 14

    # Drowsiness (PERCLOS)
    perclos = metrics.get("perclos", 0) or 0
    drowsy_pct = int(min(100, perclos * 100))
    drowsy_col = _level_color(drowsy_pct)
    _t(panel, "Drowsiness", px, y, 0.42, _GRAY)
    _t(panel, f"{drowsy_pct}%", px + bar_w + 18, y, 0.42, drowsy_col)
    y += line_gap
    _bar(panel, px, y, bar_w + 10, bar_h, perclos, drowsy_col)
    y += section_gap

    # Distraction
    is_dist = metrics.get("is_distracted", False)
    dd = metrics.get("distraction_duration_sec", 0) or 0
    if attn == "calibrating":
        dist_pct = 0
    elif is_dist:
        dist_pct = min(100, max(40, int(dd * 25)))
    else:
        dist_pct = 0
    dist_col = _level_color(dist_pct)
    _t(panel, "Distraction", px, y, 0.42, _GRAY)
    _t(panel, f"{dist_pct}%", px + bar_w + 18, y, 0.42, dist_col)
    y += line_gap
    _bar(panel, px, y, bar_w + 10, bar_h, dist_pct / 100.0, dist_col)
    y += line_gap
    _div(panel, y, w)
    y += 14

    # Eyes
    ear = metrics.get("ear", 0) or 0
    if ear > 0.25:
        eye_lbl, eye_col = "Open", _GREEN
    elif ear > 0.18:
        eye_lbl, eye_col = "Half-Open", _YELLOW
    else:
        eye_lbl, eye_col = "Closed", _RED
    _t(panel, "Eyes", px, y, 0.42, _GRAY)
    _t(panel, eye_lbl, px + 55, y, 0.42, eye_col)
    y += line_gap
    _bar(panel, px, y, bar_w + 10, bar_h, min(ear / 0.35, 1.0), eye_col)
    y += section_gap

    # Mouth
    mar = metrics.get("mar", 0) or 0
    if mar > 0.5:
        m_lbl, m_col = "Yawning", _RED
    elif mar > 0.35:
        m_lbl, m_col = "Open", _YELLOW
    else:
        m_lbl, m_col = "Closed", _GREEN
    _t(panel, "Mouth", px, y, 0.42, _GRAY)
    _t(panel, m_lbl, px + 55, y, 0.42, m_col)
    y += line_gap

    # Blinks
    blink_hz = metrics.get("blink_rate_hz", 0) or 0
    blinks = metrics.get("blink_count", 0) or 0
    _t(panel, "Blinks", px, y, 0.42, _GRAY)
    _t(panel, f"{blink_hz:.1f}/s  ({blinks})", px + 55, y, 0.40, _LIGHT)
    y += line_gap

    # Fatigue / closure warnings
    if metrics.get("fatigue_active", False):
        _t(panel, "! Driver is getting tired", px, y, 0.44, _RED, 2)
        y += line_gap
    closure = metrics.get("eye_closure_duration_sec", 0) or 0
    if closure > 0.5:
        _t(panel, f"! Eyes closed {closure:.1f}s", px, y, 0.44, _RED, 2)
        y += line_gap

    _div(panel, y, w)
    y += 14

    # Head pose
    pitch = metrics.get("pitch", 0)
    yaw = metrics.get("yaw", 0)
    roll = metrics.get("roll", 0)
    _t(panel, "Head Pose", px, y, 0.42, _GRAY)
    _t(panel, f"P{pitch:+.0f}  Y{yaw:+.0f}  R{roll:+.0f}", px + 90, y, 0.40, _LIGHT)
    y += line_gap

    # Distraction analysis
    if attn != "calibrating":
        dev_col = _RED if is_dist else _GREEN

        hdy = metrics.get("head_deviation_yaw", 0)
        hdp = metrics.get("head_deviation_pitch", 0)
        _t(panel, "Head Dev", px, y, 0.40, _GRAY)
        _t(panel, f"Y{hdy:+.0f}  P{hdp:+.0f}", px + 80, y, 0.40, dev_col)
        y += line_gap

        gdh = metrics.get("gaze_deviation_h", 0)
        gdv = metrics.get("gaze_deviation_v", 0)
        _t(panel, "Gaze Dev", px, y, 0.40, _GRAY)
        _t(panel, f"H{gdh:+.3f}  V{gdv:+.3f}", px + 80, y, 0.40, dev_col)
        y += line_gap

        align = metrics.get("alignment_score", 1.0)
        _t(panel, "Alignment", px, y, 0.40, _GRAY)
        _t(panel, f"{align:.2f}", px + 80, y, 0.40, _LIGHT)
        y += line_gap

        votes = metrics.get("votes", {})
        if votes:
            rv = votes.get("rule_based", "?")
            cv_val = votes.get("cnn", "?")
            lv = votes.get("lstm", "?")
            fv = votes.get("final", "?")
            _t(panel, "Votes", px, y, 0.40, _GRAY)
            _t(panel, f"R:{rv[:4]}", px + 55, y, 0.36, _vote_color(rv))
            _t(panel, f"C:{cv_val[:4]}", px + 120, y, 0.36, _vote_color(cv_val))
            _t(panel, f"L:{lv[:4]}", px + 185, y, 0.36, _vote_color(lv))
            y += line_gap
            final_col = _RED if fv == "distraction" else _GREEN
            _t(panel, f"Final: {fv.upper()}", px + 55, y, 0.44, final_col, 2)
            y += line_gap

        if dd > 0.5:
            _t(panel, f"Away for {dd:.1f}s", px, y, 0.44, _RED, 2)
            y += line_gap

    # Alert banner
    alert = metrics.get("alert_type")
    if alert:
        _div(panel, y, w)
        y += 6
        banner_h = 38
        cv2.rectangle(panel, (8, y), (w - 8, y + banner_h), _ALERT_BG, -1)
        msg = _alert_message(alert)
        sz = cv2.getTextSize(msg, _FONT, 0.65, 2)[0]
        _t(panel, msg, (w - sz[0]) // 2, y + 26, 0.65, (255, 255, 255), 2)

    return panel


# ── Public renderer class ─────────────────────────────────────────────
class OverlayRenderer:
    """Composes the final image: face-annotated video + dashboard panel."""

    def draw_driver_hud(self, frame, landmarks=None, img_w=None, img_h=None, **metrics):
        """Draw face annotations on *frame* in-place and return composite (frame+panel).

        Returns the hstacked numpy array (video | panel).
        """
        if landmarks is not None and img_w and img_h:
            _draw_face_overlays(frame, landmarks, img_w, img_h)

        panel = _build_panel(metrics, frame.shape[0])
        return np.hstack([frame, panel])
