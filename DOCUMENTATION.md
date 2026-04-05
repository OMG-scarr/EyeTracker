# EyeTracker — Technical Documentation

This document explains every significant piece of the EyeTracker codebase.
It is intended as a reference for future development, onboarding new
contributors, or simply refreshing your own memory months from now.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File Descriptions](#2-file-descriptions)
3. [MediaPipe Face Mesh — Landmark Map](#3-mediapipe-face-mesh--landmark-map)
4. [Eye Aspect Ratio (EAR) — Blink Detection](#4-eye-aspect-ratio-ear--blink-detection)
5. [Mouse Movement Pipeline](#5-mouse-movement-pipeline)
6. [Configuration & CLI Arguments](#6-configuration--cli-arguments)
7. [Keyboard Controls](#7-keyboard-controls)
8. [Error Handling Strategy](#8-error-handling-strategy)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)
10. [Dependency Notes](#10-dependency-notes)

---

## 1. Architecture Overview

The project follows a simple two-module design:

```
main.py      Core logic: CLI parsing, EyeTracker class, tracking loop
config.py    All tuneable constants, landmark indices, and defaults
```

`main.py` is the entry point.  It parses command-line arguments, creates an
`EyeTracker` instance, and calls `tracker.run()` which blocks until the
user quits.

`config.py` is a pure-data module with no imports.  Every magic number
lives here so you never have to hunt through logic code to change a
threshold or landmark index.

### Data flow (per frame)

```
Webcam frame
    → Mirror (cv2.flip)
    → Convert BGR→RGB
    → Wrap as mp.Image (RGB)
    → MediaPipe FaceLandmarker (.detect_for_video)
    → Extract iris landmarks 474-477  → compute screen coords → move mouse
    → Extract eye contour landmarks   → compute EAR             → click if blink
    → Draw HUD overlay on frame
    → Display in OpenCV window
    → Read keyboard input → adjust settings or quit
```

---

## 2. File Descriptions

### main.py

| Section | Purpose |
|---------|---------|
| `_eye_aspect_ratio()` | Standalone helper that computes EAR from 6 landmarks |
| `ensure_model()` | Downloads the FaceLandmarker `.task` model if not cached locally |
| `EyeTracker.__init__` | Stores settings, downloads model, creates FaceLandmarker and PyAutoGUI |
| `EyeTracker._open_camera` | Opens webcam, validates it returns frames |
| `EyeTracker._cleanup` | Releases camera, closes landmarker, and destroys OpenCV windows |
| `EyeTracker._process_frame` | Per-frame pipeline: face mesh → mouse + blink |
| `EyeTracker._move_mouse` | Converts iris position to mouse delta, calls PyAutoGUI |
| `EyeTracker._try_click` | Fires a click respecting the cooldown timer |
| `EyeTracker._draw_hud` | Renders sensitivity, threshold, and status text |
| `EyeTracker._handle_key` | Maps key presses to actions (quit, adjust, window) |
| `EyeTracker.run` | Main loop that ties everything together |
| `parse_args()` | argparse CLI definition |
| `main()` | Wires CLI args → EyeTracker → run |

### config.py

Grouped into sections:

| Section | What it contains |
|---------|-----------------|
| Landmark indices | `IRIS_LANDMARKS_SLICE`, `IRIS_TRACK_INDEX`, `LEFT_EYE_EAR_INDICES`, `RIGHT_EYE_EAR_INDICES`, `LEFT_EYE_OVERLAY` |
| Mouse movement | `DEFAULT_SENSITIVITY`, `SENSITIVITY_STEP`, `MIN_SENSITIVITY` |
| Blink detection | `DEFAULT_EAR_THRESHOLD`, `EAR_THRESHOLD_STEP`, `MIN_EAR_THRESHOLD`, `CLICK_COOLDOWN_SEC` |
| Camera & display | `DEFAULT_CAMERA_INDEX`, `WINDOW_NAME`, colour constants |

---

## 3. MediaPipe Face Mesh — Landmark Map

The MediaPipe FaceLandmarker model produces 478 landmarks per detected
face, including iris landmarks.  The first 468 are the original mesh;
landmarks 468-477 are the refined iris model.

**API note:** This project uses the modern **MediaPipe Tasks API**
(`mp.tasks.vision.FaceLandmarker`) rather than the legacy `mp.solutions`
API, which was removed in mediapipe 0.10.31+.  The Tasks API requires a
downloaded `.task` model file (auto-downloaded on first run to `models/`)
and uses `detect_for_video()` with monotonic timestamps instead of the
old `.process()` method.

### Iris landmarks (474-477)

```
474  ●─── right edge of iris (from the mesh's perspective)
475  ●─── top of iris            ← We track this one (IRIS_TRACK_INDEX = 1)
476  ●─── left edge of iris
477  ●─── bottom of iris
```

We slice `landmark[474:478]` and use index 1 within that slice (landmark
475) as the primary tracking point because vertical iris movement
correlates well with gaze direction.

### Eye contour landmarks (for EAR)

Each eye is represented by 6 points:

```
         P2         P3
          \       /
    P1 ●───●─────●───● P4
          /       \
         P6         P5
```

**Left eye** (viewer's left = person's right):
`[362, 385, 387, 263, 373, 380]`

**Right eye** (viewer's right = person's left):
`[33, 160, 158, 133, 153, 144]`

**Overlay dots** (simple two-point pair from the original code):
`[145, 159]` — upper and lower eyelid on the left eye.  These are drawn
as cyan circles purely for visual feedback and are NOT used in the EAR
calculation.

---

## 4. Eye Aspect Ratio (EAR) — Blink Detection

The previous version used a simple vertical distance between two points
(`abs(landmark[145].y - landmark[159].y)`).  This was fragile because:

- It only checked one eye.
- Raw pixel distance varies with how far you are from the camera.
- Head tilt skews the vertical measurement.

The replacement uses the **Eye Aspect Ratio** formula from:

> Soukupová, T. & Čech, J. (2016). "Real-Time Eye Blink Detection using
> Facial Landmarks."

```
EAR = (|P2-P6| + |P3-P5|) / (2 × |P1-P4|)
```

### Why EAR is better

- **Scale-invariant**: dividing vertical distances by horizontal distance
  normalises for camera distance.
- **Tilt-resistant**: because both numerator and denominator scale
  similarly under moderate head rotation.
- **Both eyes**: we compute EAR for each eye independently and average
  them, catching blinks regardless of which eye closes.

### Typical EAR values

| State | EAR range |
|-------|-----------|
| Eyes wide open | 0.30 – 0.40 |
| Normal open | 0.25 – 0.35 |
| Partial blink | 0.15 – 0.25 |
| Full blink | < 0.15 |

The default threshold is **0.21**, chosen as a middle ground.  Users
should adjust with `[` / `]` keys while watching the live EAR readout in
the top-right corner of the window.

### Click cooldown

A cooldown of `CLICK_COOLDOWN_SEC = 0.4 s` prevents a single slow blink
from firing multiple clicks.  The previous code used `pyautogui.sleep(0.2)`
which blocked the entire loop — the new approach uses a non-blocking
timestamp comparison.

---

## 5. Mouse Movement Pipeline

### Coordinate mapping

1. MediaPipe returns landmark coordinates as floats in [0, 1] relative to
   the frame.
2. We scale to pixel coordinates within the frame:
   `px = landmark.x × frame_width`.
3. We then map frame pixels to screen pixels:
   `mx = (screen_width / frame_width) × px`.
4. On the first frame we jump to the absolute position (`moveTo`).
5. On subsequent frames we compute the delta from the previous position
   and apply the sensitivity multiplier:
   `dx = (mx - prev_mx) × sensitivity`.
6. `pyautogui.moveRel(dx, dy)` shifts the system cursor.

### Why relative movement?

Absolute mapping (`moveTo`) causes the cursor to snap around because
small head movements map to large screen jumps.  Relative mode
(`moveRel`) makes the cursor feel like a trackpad — small head movements
produce proportional, smooth cursor shifts.

### Sensitivity

The `sensitivity` multiplier defaults to 2.0.  Higher values mean less
head movement is needed to cover the screen.  Users can adjust in real
time with `+` / `-`.

---

## 6. Configuration & CLI Arguments

### Runtime arguments

```
python main.py --camera 1 --sensitivity 3.0 --ear-threshold 0.18 --verbose
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--camera` | int | 0 | Webcam device index |
| `--sensitivity` | float | 2.0 | Mouse movement multiplier |
| `--ear-threshold` | float | 0.21 | Blink EAR threshold |
| `--verbose` | flag | off | Enable DEBUG logging |

All defaults are sourced from `config.py`, so editing that file changes
the baseline behaviour without touching `main.py`.

---

## 7. Keyboard Controls

All controls are active while the OpenCV window has focus:

| Key | Action |
|-----|--------|
| `+` or `=` | Increase sensitivity |
| `-` | Decrease sensitivity (min 0.1) |
| `]` | Increase EAR threshold (less sensitive blink) |
| `[` | Decrease EAR threshold (more sensitive blink) |
| `m` | Minimize window |
| `x` | Maximise / fullscreen |
| `c` or `Esc` | Quit |

---

## 8. Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| Model not downloaded | `ensure_model()` auto-downloads; if download fails, `RuntimeError` with manual download instructions |
| Camera not found | `RuntimeError` with clear message; caught in `main()`, exits with code 1 |
| Camera opens but returns no frames | Same `RuntimeError` path |
| Frame drops mid-session | `logger.warning` + `continue` (retries next iteration) |
| No face detected | "No face detected" overlay on frame; mouse stays still |
| Division by zero in EAR | Guarded: returns 0.0 if horizontal eye distance is zero |
| Ctrl+C | Caught by `KeyboardInterrupt`; cleanup runs via `finally` |
| Any other crash | `finally` block ensures camera is released and windows closed |

---

## 9. Known Limitations & Future Work

### Current limitations

- **Single-face only**: `max_num_faces=1` — won't work if multiple people
  are in frame (it picks the first detected).
- **Platform-specific window controls**: `cv2.setWindowProperty` for
  minimise/maximise behaves differently across Linux, macOS, and Windows.
- **No scroll support**: blinks trigger clicks but there's no gesture for
  scrolling yet.
- **Lighting dependent**: MediaPipe accuracy drops significantly in low
  light.

### Ideas for future versions

- **Dwell-click**: hold gaze on a spot for N seconds to click, as an
  alternative to blink-click.
- **Right-click gesture**: e.g., wink with right eye only.
- **Scroll gesture**: e.g., tilt head up/down past a threshold.
- **Smoothing filter**: apply exponential moving average to cursor
  position to reduce jitter.
- **Calibration wizard**: on first run, guide the user through looking at
  screen corners to build a personalised coordinate map.
- **System tray mode**: run in the background with a tray icon instead of
  a visible OpenCV window.
- **Configuration file**: support a `config.yaml` or `.env` file for
  persistent user preferences.
- **Tests**: add unit tests for `_eye_aspect_ratio`, coordinate mapping,
  and key handling.

---

## 10. Dependency Notes

| Package | Role | Version guidance |
|---------|------|-----------------|
| **mediapipe** | FaceLandmarker model for 478-landmark detection (Tasks API) | ≥ 0.10.0 (uses the new Tasks API, compatible with all versions including 0.10.31+) |
| **opencv-python** | Webcam capture, image processing, window display | ≥ 4.8.0 |
| **pyautogui** | Cross-platform mouse control (move, click) | ≥ 0.9.54 |

The FaceLandmarker `.task` model (~5 MB) is automatically downloaded on
first run and cached in the `models/` directory (git-ignored).  MediaPipe
bundles a TensorFlow Lite runtime internally.  No GPU is required —
inference runs on CPU and is fast enough for real-time use at 30 fps on
most modern machines.

PyAutoGUI's `FAILSAFE` feature (moving the mouse to a screen corner to
abort) is deliberately disabled in this project because an eye-controlled
cursor may drift to corners unintentionally.  If you need an emergency
stop, press `Esc` or `Ctrl+C` in the terminal.

---

*Last updated: April 2026*
