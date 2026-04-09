"""
config.py — Default configuration and constants for EyeTracker.

This module centralises every tuneable parameter so that main.py stays
focused on logic.  Values here act as defaults; they can be overridden
at runtime via command-line arguments (see main.py --help).
"""

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/
#            modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ---------------------------------------------------------------------------

# Iris tracking landmarks (right iris in the mesh, but appears as the
# left iris on a mirrored/selfie image).
# 474 = iris centre-right, 475 = iris top, 476 = iris centre-left, 477 = iris bottom.
# We use index 1 of this slice (landmark 475) as the primary gaze anchor
# because it sits at the top of the iris and moves most visibly with gaze.
IRIS_LANDMARKS_SLICE = slice(474, 478)
IRIS_TRACK_INDEX = 1  # Index within the slice used for mouse positioning

# Eye Aspect Ratio (EAR) landmarks — used for blink detection.
# Each eye is described by 6 landmarks that outline its contour:
#   P1 (outer corner), P2 (upper-outer), P3 (upper-inner),
#   P4 (inner corner), P5 (lower-inner), P6 (lower-outer).
#
# Left eye (from the viewer's perspective, right eye on the person):
LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye (from the viewer's perspective, left eye on the person):
RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]

# Simple two-point eye landmarks kept for the on-screen overlay circles.
LEFT_EYE_OVERLAY = [145, 159]

# ---------------------------------------------------------------------------
# Mouse movement defaults
# ---------------------------------------------------------------------------
DEFAULT_SENSITIVITY = 2.0       # Multiplier applied to relative mouse deltas
SENSITIVITY_STEP = 0.2          # Amount added/subtracted per key press
MIN_SENSITIVITY = 0.1           # Floor so the mouse never "freezes"

# ---------------------------------------------------------------------------
# Blink / click detection defaults
# ---------------------------------------------------------------------------
DEFAULT_EAR_THRESHOLD = 0.21    # EAR below this value counts as a blink
EAR_THRESHOLD_STEP = 0.005      # Adjustment per key press
MIN_EAR_THRESHOLD = 0.05        # Practical minimum
CLICK_COOLDOWN_SEC = 0.4        # Minimum seconds between consecutive clicks

# ---------------------------------------------------------------------------
# Camera & display
# ---------------------------------------------------------------------------
DEFAULT_CAMERA_INDEX = 0
WINDOW_NAME = "EyeTracker"

# Compact window size (Apple-style miniaturised)
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 360

# Legacy on-screen text colours (BGR) — kept for any fallback usage
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
