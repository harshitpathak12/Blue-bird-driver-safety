import cv2
import numpy as np
import time
from collections import deque

from utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds and window sizes inspired by the notebook Model3 implementation
EAR_THRESHOLD = 0.22
PERCLOS_WINDOW = 2.0  # seconds window
DROWSY_THRESHOLD = 0.4
# Very low blink rate (blinks per second) can be an additional fatigue signal
BLINK_RATE_LOW_THRESHOLD = 0.1

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def compute_ear(eye):
    p1, p2, p3, p4, p5, p6 = eye
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float((vertical1 + vertical2) / (2.0 * horizontal))


class ModelDrowsiness:
    """
    Drowsiness model using PERCLOS and blink counting.

    Unlike the notebook version, this runtime variant does not perform its own
    face landmark detection. It expects MediaPipe landmarks and image size to
    be passed in (typically from a shared FaceDetector).
    """

    def __init__(self):
        self.blink_count = 0
        self.frame_counter = 0
        self.closed_frames = deque()
        self.perclos = 0.0

        # Blink rate over the sliding window
        self.blink_rate_hz = 0.0
        self.blink_rate_low = False
        self._blink_timestamps = deque()

        # Continuous eyelid closure duration for sleep detection
        self.eye_closure_duration_sec = 0.0
        self._eye_closed_start = None

    def process(self, frame, landmarks, img_w, img_h):
        """
        Update PERCLOS / blink statistics and overlay them on the frame.

        If landmarks are None, metrics are left unchanged and the frame is
        returned as-is.
        """
        current_time = time.time()
        eyes_closed = False

        if landmarks is not None and img_w is not None and img_h is not None:
            h, w, _ = frame.shape

            left_eye = np.array(
                [[landmarks[i][0] * w, landmarks[i][1] * h] for i in LEFT_EYE]
            )
            right_eye = np.array(
                [[landmarks[i][0] * w, landmarks[i][1] * h] for i in RIGHT_EYE]
            )

            left_ear = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                self.frame_counter += 1
                eyes_closed = True
            else:
                if self.frame_counter >= 2:
                    # Register a completed blink
                    self.blink_count += 1
                    self._blink_timestamps.append(current_time)
                self.frame_counter = 0

        # Update eye-closure duration (continuous closure timer)
        if eyes_closed:
            if self._eye_closed_start is None:
                self._eye_closed_start = current_time
            self.eye_closure_duration_sec = current_time - self._eye_closed_start
        else:
            self._eye_closed_start = None
            self.eye_closure_duration_sec = 0.0

        # Update PERCLOS sliding window
        self.closed_frames.append((current_time, eyes_closed))
        while self.closed_frames and (
            current_time - self.closed_frames[0][0] > PERCLOS_WINDOW
        ):
            self.closed_frames.popleft()

        if self.closed_frames:
            closed_count = sum(1 for _, closed in self.closed_frames if closed)
            self.perclos = closed_count / len(self.closed_frames)

        # Maintain blink timestamps within the same window and compute rate
        while self._blink_timestamps and (
            current_time - self._blink_timestamps[0] > PERCLOS_WINDOW
        ):
            self._blink_timestamps.popleft()

        if self._blink_timestamps:
            self.blink_rate_hz = len(self._blink_timestamps) / PERCLOS_WINDOW
            self.blink_rate_low = self.blink_rate_hz < BLINK_RATE_LOW_THRESHOLD
        else:
            self.blink_rate_hz = 0.0
            self.blink_rate_low = False

        # HUD (PERCLOS, Blinks, alerts) drawn by app.utils.overlay
        return frame

