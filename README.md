# EyeTracker

A Python application that uses your webcam and MediaPipe to control your mouse pointer and perform clicks using eye and head movement.

## Features
- Move mouse pointer with head movement
- Click by blinking (distance between eye landmarks)
- Adjustable mouse sensitivity (press `+`/`-`)
- Adjustable click threshold (press `[`/`]`)
- Minimize, maximize, and close the window with keyboard shortcuts (`m`, `x`, `c`)
- Visual feedback for landmarks and settings

## Requirements
- Python 3.7–3.10
- mediapipe
- opencv-python
- pyautogui

## Installation
1. Clone this repository or copy the files to your project folder.
2. Install dependencies:
   ```
   pip install mediapipe opencv-python pyautogui
   ```
3. Run the program:
   ```
   python main.py
   ```

## Usage
- Move your head to control the mouse pointer.
- Blink (close left eye) to click.
- Adjust sensitivity with `+`/`-` keys.
- Adjust click threshold with `[`/`]` keys.
- Minimize window: `m`, Maximize: `x`, Close: `c`.
- Press `Esc` to exit.

## Notes
- For best results, use in a well-lit environment.
- Adjust sensitivity and click threshold to your comfort.
- Works best with a 1920x1080 screen and a standard webcam.

## License
MIT
