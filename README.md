# ✋ Magic Hand — Virtual Mouse

> Control your mouse with nothing but your hand. No hardware required.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange?style=flat-square)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)



---

## What it does

Magic Hand turns your webcam into a gesture-driven virtual mouse. A real-time hand-tracking pipeline detects 21 landmark positions at up to 30 fps, maps your index fingertip to screen coordinates, and fires left/right/double clicks and momentum scroll from pinch gestures.

---

## Gestures

| Gesture | Action |
|---|---|
| 👆🤙 Thumb + Index pinch | Left Click |
| 🤙🖕 Thumb + Middle pinch | Double Click |
| ☝️✌️ Index + Middle pinch | Right Click |
| ✌️↕️ Spread Index + Middle, move up/down | Scroll |

---

## Requirements

- Python 3.8+
- Webcam (720p recommended)

```
opencv-python
mediapipe
pyautogui
numpy
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/magic-hand.git
cd magic-hand

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install opencv-python mediapipe pyautogui numpy

# 4. Run
python magic_hand.py
```

---

## How it Works

1. **Calibration** — Hold your open hand in frame for 3 seconds. The script measures your hand span and computes adaptive pinch/scroll thresholds personal to your hand size and camera distance.
2. **Landmark Detection** — Each frame is passed to MediaPipe Hands, returning 21 3D landmarks. Only the most confident single hand is tracked (`min_confidence ≥ 0.75`).
3. **Cursor Mapping** — Index fingertip position is interpolated from the camera's active zone to full screen coordinates, then smoothed.
4. **Gesture Recognition** — Euclidean distances between landmark pairs are compared against adaptive thresholds. Gestures require `HOLD_FRAMES` consecutive qualifying frames to fire, eliminating accidental triggers.
5. **Momentum Scroll** — When index + middle are spread wide, the vertical delta is buffered over 5 frames to produce a smooth, weighted scroll.
6. **Adaptive Recalibration** — A rolling 60-frame window continuously updates thresholds as you move closer or farther from the camera.

---

## Configuration

All tuning constants are at the top of `magic_hand.py`:

| Parameter | Default | Description |
|---|---|---|
| `SMOOTHENING` | `2` | Cursor lag — higher = smoother but slower |
| `FRAME_REDUCTION` | `80` | Pixel border inset of the tracking zone |
| `CLICK_DELAY` | `0.4s` | Minimum seconds between click events |
| `HOLD_FRAMES` | `4` | Consecutive frames a pinch must hold to fire |
| `SCROLL_SPEED` | `4` | Multiplier applied to scroll delta |
| `CALIBRATION_SECS` | `3` | Duration of startup calibration |
| `MIN_HAND_CONFIDENCE` | `0.75` | MediaPipe confidence threshold |
| `PINCH_RATIO` | `0.13` | Pinch distance as fraction of hand span |
| `SCROLL_RATIO` | `0.22` | Spread distance that activates scroll mode |

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` / `Esc` | Quit |
| `C` | Recalibrate |

---

## Tips

- **Lighting** — Even, front-facing light significantly improves landmark accuracy. Avoid strong backlight.
- **Distance** — Keep your hand 40–70 cm from the camera.
- **Accidental clicks** — Increase `HOLD_FRAMES` to 6–8 if gestures fire too easily.
- **Slow cursor** — Lower `SMOOTHENING` to `1` for a snappier 1:1 feel.
- **Recalibrate** — Press `C` whenever you reposition or lighting changes.

---

## License

MIT — do whatever you like, just keep the attribution.
