"""
ui.py — Apple Camera-style HUD overlay for EyeTracker.

Renders a sleek, miniaturised camera interface inspired by iOS/macOS camera
apps: translucent dark bars, pill-shaped indicators, corner viewfinder
brackets, and minimal typography.
"""

import cv2
import numpy as np
import time

import config as cfg


# ──────────────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────────────

def _overlay_rect(frame, x, y, w, h, color, alpha):
    """Draw a semi-transparent filled rectangle (alpha 0..1)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _rounded_rect(frame, x, y, w, h, radius, color, alpha):
    """Draw a semi-transparent rounded rectangle."""
    overlay = frame.copy()
    # Four corners
    cv2.circle(overlay, (x + radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1)
    # Fill center rects
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _pill(frame, cx, cy, w, h, color, alpha):
    """Draw a pill / capsule shape centered at (cx, cy)."""
    r = h // 2
    x = cx - w // 2
    y = cy - h // 2
    _rounded_rect(frame, x, y, w, h, r, color, alpha)


def _put_text_centered(frame, text, cx, cy, scale, color, thickness=1):
    """Put text centered on (cx, cy)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (cx - tw // 2, cy + th // 2)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _put_text(frame, text, x, y, scale, color, thickness=1):
    """Put text at (x, y) baseline-left."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# Viewfinder corner brackets
# ──────────────────────────────────────────────────────────────────────

def _draw_corner_brackets(frame, rect, color, thickness=2, length=20):
    """Draw L-shaped corner brackets around a rectangle (x, y, w, h)."""
    x, y, w, h = rect
    # Top-left
    cv2.line(frame, (x, y), (x + length, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x, y + length), color, thickness, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# Apple-style HUD class
# ──────────────────────────────────────────────────────────────────────

class AppleHUD:
    """Renders an Apple Camera-inspired HUD on each video frame."""

    # Color palette (BGR)
    BAR_BG = (20, 20, 20)
    PILL_BG = (50, 50, 50)
    PILL_ACTIVE = (60, 60, 60)
    WHITE = (255, 255, 255)
    GRAY = (160, 160, 160)
    DIM = (100, 100, 100)
    GREEN = (100, 220, 100)
    RED = (80, 80, 240)
    AMBER = (50, 180, 255)
    ACCENT = (250, 200, 60)       # warm gold accent
    BRACKET_COLOR = (255, 255, 255)

    TOP_BAR_H = 44
    BOTTOM_BAR_H = 72

    def __init__(self):
        self._face_pulse = 0.0     # for pulsing face indicator
        self._last_click_t = 0.0   # flash on click
        self._click_flash = False

    def notify_click(self):
        """Call when a blink-click fires so the HUD can flash."""
        self._last_click_t = time.time()
        self._click_flash = True

    # ---- main entry ----------------------------------------------------------

    def draw(self, frame, face_detected, sensitivity, ear_threshold,
             avg_ear=None, tracking_active=True):
        """Render the full Apple-style HUD onto *frame* (mutates in place)."""
        fh, fw = frame.shape[:2]

        # -- translucent top bar -----------------------------------------------
        _overlay_rect(frame, 0, 0, fw, self.TOP_BAR_H, self.BAR_BG, 0.65)
        self._draw_top_bar(frame, fw, face_detected, avg_ear)

        # -- translucent bottom bar --------------------------------------------
        by = fh - self.BOTTOM_BAR_H
        _overlay_rect(frame, 0, by, fw, self.BOTTOM_BAR_H, self.BAR_BG, 0.65)
        self._draw_bottom_bar(frame, fw, fh, sensitivity, ear_threshold)

        # -- viewfinder brackets (centered, covering face detection zone) -------
        margin_x = int(fw * 0.18)
        margin_top = self.TOP_BAR_H + 12
        margin_bot = self.BOTTOM_BAR_H + 12
        bracket_rect = (
            margin_x,
            margin_top,
            fw - 2 * margin_x,
            fh - margin_top - margin_bot,
        )
        bracket_col = self.GREEN if face_detected else self.DIM
        _draw_corner_brackets(frame, bracket_rect, bracket_col, 2, 22)

        # -- click flash (brief white vignette) --------------------------------
        if self._click_flash:
            elapsed = time.time() - self._last_click_t
            if elapsed < 0.12:
                alpha = max(0, 0.25 * (1 - elapsed / 0.12))
                _overlay_rect(frame, 0, 0, fw, fh, (255, 255, 255), alpha)
            else:
                self._click_flash = False

        # -- no-face warning ---------------------------------------------------
        if not face_detected:
            _pill(frame, fw // 2, fh // 2, 180, 32, self.BAR_BG, 0.7)
            _put_text_centered(frame, "No Face Detected", fw // 2, fh // 2,
                               0.45, self.AMBER, 1)

    # ---- top bar -------------------------------------------------------------

    def _draw_top_bar(self, frame, fw, face_detected, avg_ear):
        cy = self.TOP_BAR_H // 2

        # Left: face-detection status dot + label
        dot_color = self.GREEN if face_detected else self.RED
        cv2.circle(frame, (18, cy), 5, dot_color, -1, cv2.LINE_AA)
        label = "Tracking" if face_detected else "Searching"
        _put_text(frame, label, 30, cy + 4, 0.38, self.GRAY, 1)

        # Center: app name pill
        _pill(frame, fw // 2, cy, 110, 24, self.PILL_BG, 0.5)
        _put_text_centered(frame, "EyeTracker", fw // 2, cy, 0.38, self.WHITE, 1)

        # Right: live EAR readout
        if avg_ear is not None:
            ear_str = f"EAR {avg_ear:.3f}"
            ear_col = self.GREEN if avg_ear >= 0.21 else self.RED
            _put_text(frame, ear_str, fw - 110, cy + 4, 0.36, ear_col, 1)

    # ---- bottom bar ----------------------------------------------------------

    def _draw_bottom_bar(self, frame, fw, fh, sensitivity, ear_threshold):
        cy = fh - self.BOTTOM_BAR_H // 2

        # Three pills spread across the bottom bar
        third = fw // 3

        # Pill 1 — Sensitivity
        _pill(frame, third // 2, cy, max(100, third - 20), 30, self.PILL_BG, 0.55)
        _put_text_centered(frame, f"Sens {sensitivity:.1f}",
                           third // 2, cy - 1, 0.36, self.WHITE, 1)

        # Pill 2 — Threshold
        _pill(frame, fw // 2, cy, max(100, third - 20), 30, self.PILL_BG, 0.55)
        _put_text_centered(frame, f"Blink {ear_threshold:.3f}",
                           fw // 2, cy - 1, 0.36, self.WHITE, 1)

        # Pill 3 — Shortcuts hint
        _pill(frame, fw - third // 2, cy, max(100, third - 20), 30, self.PILL_BG, 0.55)
        _put_text_centered(frame, "+/-  [/]  Esc",
                           fw - third // 2, cy - 1, 0.33, self.DIM, 1)

    # ---- iris reticle (replaces raw red dots) --------------------------------

    @staticmethod
    def draw_iris_reticle(frame, cx, cy, radius=8):
        """Draw a subtle crosshair reticle at the iris position."""
        col = (255, 255, 255)
        r = radius
        # Thin cross lines
        cv2.line(frame, (cx - r, cy), (cx - 3, cy), col, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + 3, cy), (cx + r, cy), col, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - r), (cx, cy - 3), col, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 3), (cx, cy + r), col, 1, cv2.LINE_AA)
        # Small center dot
        cv2.circle(frame, (cx, cy), 2, col, -1, cv2.LINE_AA)

    # ---- eye contour (replaces cyan dots) ------------------------------------

    @staticmethod
    def draw_eye_points(frame, points):
        """Draw subtle small dots for eye landmarks."""
        for (x, y) in points:
            cv2.circle(frame, (x, y), 2, (200, 200, 200), -1, cv2.LINE_AA)
