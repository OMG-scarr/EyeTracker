# EyeTracker

Control your computer's mouse pointer and perform clicks using eye and head movement — no hands required. Built with [MediaPipe Face Mesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_mesh), OpenCV, and PyAutoGUI.

## Features

- **Head-driven cursor**: move the mouse pointer by moving your head; iris tracking provides precise gaze positioning.
- **Blink-to-click**: close your eyes briefly to click. Uses the [Eye Aspect Ratio (EAR)](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf) algorithm for robust, scale-invariant blink detection across both eyes.
- **Real-time tuning**: adjust sensitivity and click threshold on the fly with keyboard shortcuts.
- **Visual feedback**: live overlay shows landmark dots, current settings, and EAR value for easy calibration.
- **Command-line configuration**: set camera index, sensitivity, and threshold without editing code.

## Requirements

- Python 3.8 – 3.12
- A standard webcam

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EyeTracker.git
cd EyeTracker

# (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
python main.py
```

### Command-line options

```
python main.py --camera 1             # use a different webcam
python main.py --sensitivity 3.0      # faster cursor movement
python main.py --ear-threshold 0.18   # more sensitive blink detection
python main.py --verbose              # enable debug logging
python main.py --help                 # see all options
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `+` / `-` | Increase / decrease mouse sensitivity |
| `]` / `[` | Increase / decrease blink EAR threshold |
| `m` | Minimize window |
| `x` | Maximize / fullscreen |
| `c` / `Esc` | Quit |

## How It Works

1. The webcam captures your face in real time.
2. MediaPipe Face Mesh detects 478 facial landmarks, including refined iris positions.
3. Iris landmark movement is translated into relative mouse movement (like a trackpad).
4. Blink detection uses the Eye Aspect Ratio — a ratio of vertical to horizontal eye distances — which is robust to camera distance and head tilt.
5. When EAR drops below the threshold, a click is fired (with a cooldown to prevent double-clicks).

For a deep technical walkthrough, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Tips for Best Results

- Use in a **well-lit environment** — MediaPipe accuracy drops in low light.
- Start with default settings and adjust sensitivity/threshold while watching the live EAR readout (top-right corner).
- Sit at a comfortable distance from your webcam (roughly arm's length).
- If the cursor feels too fast, lower the sensitivity with `-`. If blinks aren't registering, lower the EAR threshold with `[`.

## Project Structure

```
EyeTracker/
├── main.py              # Entry point and core EyeTracker class
├── config.py            # All tuneable constants and landmark indices
├── models/              # Auto-downloaded FaceLandmarker model (git-ignored)
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Package metadata (PEP 621)
├── DOCUMENTATION.md     # In-depth technical documentation
├── LICENSE              # MIT License
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## License

[MIT](LICENSE) — Sir. Oscar MG
