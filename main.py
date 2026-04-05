"""
main.py — Entry point for EyeTracker.

An accessibility tool that lets you control your computer's mouse pointer
and perform clicks using eye / head movement captured through a standard
webcam.  Powered by the Google MediaPipe Tasks FaceLandmarker API and OpenCV.

Usage:
    python main.py                        # defaults
    python main.py --sensitivity 3.0      # faster cursor
    python main.py --camera 1             # use second webcam
    python main.py --help                 # see all options

Keyboard controls while the window is focused:
    +  / -     Increase / decrease mouse sensitivity
    [  / ]     Decrease / increase blink-click EAR threshold
    m          Minimize window
    x          Maximise / fullscreen window
    c / Esc    Quit
"""

import argparse
import logging
import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import pyautogui

import config as cfg

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eyetracker")

# ---------------------------------------------------------------------------
# MediaPipe Tasks API aliases
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def ensure_model():
    """Download the FaceLandmarker model if it isn't already cached locally."""
    if os.path.isfile(MODEL_PATH):
        logger.debug("Model found at %s", MODEL_PATH)
        return MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Downloading FaceLandmarker model (~5 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("Model saved to %s", MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download the FaceLandmarker model.\n"
            f"URL: {MODEL_URL}\n"
            f"Error: {exc}\n\n"
            f"You can download it manually and place it at:\n  {MODEL_PATH}"
        ) from exc
    return MODEL_PATH


# ---------------------------------------------------------------------------
# Helper: Eye Aspect Ratio (EAR)
# ---------------------------------------------------------------------------
def _eye_aspect_ratio(landmarks, indices):
    """Compute the Eye Aspect Ratio for one eye.

    The EAR formula (Soukupova & Cech, 2016) measures how "open" an eye is:

        EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)

    Where P1..P6 are six landmark points around the eye contour.

    Returns a float — roughly 0.25-0.35 when open, dropping below ~0.20 on
    a blink.  The exact numbers depend on the person and camera distance.
    """
    def _dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
    vertical_a = _dist(p2, p6)
    vertical_b = _dist(p3, p5)
    horizontal = _dist(p1, p4)

    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


# ---------------------------------------------------------------------------
# Core tracker class
# ---------------------------------------------------------------------------
class EyeTracker:
    """Encapsulates the full eye-tracking loop: camera -> face mesh -> mouse."""

    def __init__(self, camera_index, sensitivity, ear_threshold):
        self.camera_index = camera_index
        self.sensitivity = sensitivity
        self.ear_threshold = ear_threshold

        # Previous frame's mouse position (for relative movement)
        self._prev_mx = None
        self._prev_my = None

        # Timestamp of the last registered click (cooldown guard)
        self._last_click_time = 0.0

        # Monotonic frame counter for detect_for_video (milliseconds)
        self._frame_timestamp_ms = 0

        # Screen dimensions for coordinate mapping
        self.screen_w, self.screen_h = pyautogui.size()

        # Disable pyautogui's built-in pause and fail-safe for smoother
        # real-time control.  The fail-safe (move mouse to corner to abort)
        # can interfere with an eye-controlled cursor that naturally drifts.
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

        # Download model if needed and create FaceLandmarker
        model_path = ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Camera handle (opened lazily in .run())
        self.cap = None

    # ----- lifecycle -------------------------------------------------------

    def _open_camera(self):
        """Open the webcam and verify we get at least one frame."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                "Check that your webcam is connected and not in use by "
                "another application."
            )
        ret, _ = self.cap.read()
        if not ret:
            raise RuntimeError(
                "Camera opened but returned no frames. "
                "Try a different --camera index."
            )
        logger.info(
            "Camera %d opened  (%dx%d)",
            self.camera_index,
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _cleanup(self):
        """Release hardware resources."""
        if self.cap is not None:
            self.cap.release()
        self.landmarker.close()
        cv2.destroyAllWindows()
        logger.info("Resources released — goodbye.")

    # ----- per-frame processing --------------------------------------------

    def _process_frame(self, frame):
        """Run FaceLandmarker on *frame* and drive the mouse.

        Returns the annotated frame (for display) and a bool indicating
        whether a face was detected.
        """
        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        frame_h, frame_w, _ = frame.shape

        # Convert OpenCV BGR frame to MediaPipe Image (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Advance the monotonic timestamp for VIDEO mode
        self._frame_timestamp_ms += 33  # ~30 fps
        results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        face_detected = False
        if results.face_landmarks:
            face_detected = True
            # results.face_landmarks is a list of faces; each face is a list
            # of NormalizedLandmark objects with .x, .y, .z attributes.
            lm = results.face_landmarks[0]

            # --- Iris tracking (mouse movement) ---
            # Landmarks 474-477 are the iris landmarks (when available).
            # 474 = right edge, 475 = top, 476 = left edge, 477 = bottom.
            for idx, li in enumerate(range(474, 478)):
                if li >= len(lm):
                    break  # model didn't return iris landmarks
                point = lm[li]
                px = int(point.x * frame_w)
                py = int(point.y * frame_h)
                cv2.circle(frame, (px, py), 2, cfg.COLOR_RED, -1)

                if idx == cfg.IRIS_TRACK_INDEX:
                    mx = int(self.screen_w / frame_w * px)
                    my = int(self.screen_h / frame_h * py)
                    self._move_mouse(mx, my)

            # --- Eye overlay (cyan dots on left-eye contour) ---
            for li in cfg.LEFT_EYE_OVERLAY:
                if li < len(lm):
                    ox = int(lm[li].x * frame_w)
                    oy = int(lm[li].y * frame_h)
                    cv2.circle(frame, (ox, oy), 2, cfg.COLOR_CYAN, -1)

            # --- Blink detection via EAR ---
            left_ear = _eye_aspect_ratio(lm, cfg.LEFT_EYE_EAR_INDICES)
            right_ear = _eye_aspect_ratio(lm, cfg.RIGHT_EYE_EAR_INDICES)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.ear_threshold:
                self._try_click(avg_ear)

            # Show live EAR value for debugging / calibration
            ear_color = cfg.COLOR_GREEN if avg_ear >= self.ear_threshold else cfg.COLOR_RED
            cv2.putText(
                frame,
                f"EAR: {avg_ear:.3f}",
                (frame_w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2,
            )

        return frame, face_detected

    def _move_mouse(self, mx, my):
        """Move the system cursor based on iris position delta."""
        if self._prev_mx is not None and self._prev_my is not None:
            dx = (mx - self._prev_mx) * self.sensitivity
            dy = (my - self._prev_my) * self.sensitivity
            pyautogui.moveRel(dx, dy, _pause=False)
        else:
            # First frame — jump to absolute position
            pyautogui.moveTo(mx, my, _pause=False)
        self._prev_mx, self._prev_my = mx, my

    def _try_click(self, ear_value):
        """Fire a click if the cooldown period has elapsed."""
        now = time.time()
        if now - self._last_click_time >= cfg.CLICK_COOLDOWN_SEC:
            pyautogui.click(_pause=False)
            self._last_click_time = now
            logger.info("Click — EAR %.4f (threshold %.3f)", ear_value, self.ear_threshold)

    # ----- HUD overlay -----------------------------------------------------

    def _draw_hud(self, frame, face_detected):
        """Draw settings and status text on the video frame."""
        cv2.putText(
            frame,
            f"Sensitivity: {self.sensitivity:.1f}  [+/-]",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cfg.COLOR_WHITE, 2,
        )
        cv2.putText(
            frame,
            f"EAR Threshold: {self.ear_threshold:.3f}  [ [ / ] ]",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cfg.COLOR_WHITE, 2,
        )
        cv2.putText(
            frame,
            "Quit[c/Esc]  Min[m]  Max[x]",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.COLOR_YELLOW, 2,
        )
        if not face_detected:
            cv2.putText(
                frame,
                "No face detected",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cfg.COLOR_RED, 2,
            )

    # ----- keyboard handling ------------------------------------------------

    def _handle_key(self, key):
        """Process a single key press.  Returns True to signal quit."""
        if key == 27 or key == ord("c"):           # Esc or 'c'
            return True
        if key == ord("m"):
            cv2.setWindowProperty(cfg.WINDOW_NAME, cv2.WND_PROP_VISIBLE, 0)
        elif key == ord("x"):
            cv2.setWindowProperty(
                cfg.WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        elif key in (ord("+"), ord("=")):
            self.sensitivity += cfg.SENSITIVITY_STEP
        elif key == ord("-"):
            self.sensitivity = max(cfg.MIN_SENSITIVITY, self.sensitivity - cfg.SENSITIVITY_STEP)
        elif key == ord("]"):
            self.ear_threshold += cfg.EAR_THRESHOLD_STEP
        elif key == ord("["):
            self.ear_threshold = max(cfg.MIN_EAR_THRESHOLD, self.ear_threshold - cfg.EAR_THRESHOLD_STEP)
        return False

    # ----- main loop --------------------------------------------------------

    def run(self):
        """Start the tracking loop.  Blocks until the user quits."""
        self._open_camera()
        cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
        logger.info(
            "Tracking started — sensitivity=%.1f  ear_threshold=%.3f",
            self.sensitivity, self.ear_threshold,
        )

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Camera returned no frame — retrying...")
                    continue

                frame, face_detected = self._process_frame(frame)
                self._draw_hud(frame, face_detected)
                cv2.imshow(cfg.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key):
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        finally:
            self._cleanup()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="EyeTracker — control your mouse with eye and head movement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera", type=int, default=cfg.DEFAULT_CAMERA_INDEX,
        help="Webcam device index (0 = default camera).",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=cfg.DEFAULT_SENSITIVITY,
        help="Mouse movement multiplier.",
    )
    parser.add_argument(
        "--ear-threshold", type=float, default=cfg.DEFAULT_EAR_THRESHOLD,
        help="Eye Aspect Ratio below which a blink is registered as a click.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    tracker = EyeTracker(
        camera_index=args.camera,
        sensitivity=args.sensitivity,
        ear_threshold=args.ear_threshold,
    )

    try:
        tracker.run()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
