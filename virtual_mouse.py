import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# ══════════════════════════════════════════════════════
#  MAGIC HAND — Virtual Mouse  (v3)
#  Fixes: confidence filtering, adaptive thresholds,
#         gesture calibration, smooth scroll
# ══════════════════════════════════════════════════════

# ── Tunable constants ──────────────────────────────────
SMOOTHENING         = 2       # 1 = raw, 5 = very smooth
FRAME_REDUCTION     = 80      # active-zone border (px)
CLICK_DELAY         = 0.4     # min seconds between clicks
HOLD_FRAMES         = 3       # frames a pinch must hold before firing
SCROLL_HOLD_FRAMES  = 2       # frames before scroll starts
SCROLL_SPEED        = 4       # scroll units per trigger
CALIBRATION_SECS    = 3       # seconds for calibration phase
MIN_HAND_CONFIDENCE = 0.75    # drop frames below this confidence

# ── Adaptive threshold defaults ────────────────────────
ADAPT_WINDOW        = 60      # frames to compute hand size reference
PINCH_RATIO         = 0.12    # pinch distance / hand_span to trigger
SCROLL_RATIO        = 0.20    # index-middle spread / hand_span for scroll

# ──────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    model_complexity=0
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# ── State ──────────────────────────────────────────────
prev_x, prev_y     = 0, 0
last_click_time    = 0
p_time             = 0

# Pinch hold counters
ti_frames = tm_frames = im_frames = 0

# Scroll state (smooth momentum)
scroll_buffer  = deque(maxlen=5)
last_scroll_y  = None
scroll_hold    = 0

# Adaptive threshold engine
hand_span_history = deque(maxlen=ADAPT_WINDOW)
adaptive_pinch    = 35.0
adaptive_scroll   = 60.0

# Calibration state
calibrating   = True
calib_start   = time.time()
calib_spans   = []

# ── Helpers ────────────────────────────────────────────
def dist(p1, p2):
    return float(np.hypot(p1[0]-p2[0], p1[1]-p2[1]))

def hand_span(lm):
    """Wrist to middle MCP (lm0 to lm9) — stable size reference."""
    return dist(lm[0], lm[9])

def draw_indicator(img, p1, p2, ratio, threshold_ratio):
    """Line between two fingertips, colour shifts as ratio approaches threshold."""
    t = min(ratio / max(threshold_ratio, 1e-6), 1.0)
    r = int(200 * (1 - t))
    g = int(50 + 205 * t)
    color = (0, g, r)
    cv2.line(img, p1, p2, color, 2)
    cv2.circle(img, p1, 7, color, cv2.FILLED)
    cv2.circle(img, p2, 7, color, cv2.FILLED)

def overlay_text(img, text, pos, scale=0.8, color=(0,255,0), thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        cv2.rectangle(img, (x-4, y-th-4), (x+tw+4, y+6), (0,0,0), -1)
    cv2.putText(img, text, pos, font, scale, color, thickness)

def draw_progress_bar(img, x, y, w, h, value, max_val, color):
    cv2.rectangle(img, (x, y), (x+w, y+h), (60,60,60), -1)
    filled = int(w * min(value / max(max_val, 1), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x+filled, y+h), color, -1)

# ══════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════
while True:
    success, img = cap.read()
    if not success:
        break

    img  = cv2.flip(img, 1)
    h, w = img.shape[:2]
    FR   = FRAME_REDUCTION

    # Active zone rectangle
    cv2.rectangle(img, (FR, FR), (w-FR, h-FR), (255, 0, 255), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture_text  = ""

    # ── CALIBRATION PHASE ─────────────────────────────
    if calibrating:
        elapsed   = time.time() - calib_start
        remaining = CALIBRATION_SECS - elapsed
        if remaining > 0:
            msg = f"Open your hand fully — calibrating {remaining:.1f}s"
            overlay_text(img, msg, (FR+10, h//2 - 20),
                         scale=0.9, color=(0,255,255), thickness=2)
            overlay_text(img, "Press  C  anytime to recalibrate",
                         (FR+10, h//2 + 20),
                         scale=0.6, color=(180,180,180), thickness=1)
            if results.multi_hand_landmarks and results.multi_handedness:
                conf = results.multi_handedness[0].classification[0].score
                if conf >= MIN_HAND_CONFIDENCE:
                    lm = [(int(lk.x*w), int(lk.y*h))
                          for lk in results.multi_hand_landmarks[0].landmark]
                    calib_spans.append(hand_span(lm))
                    mp_draw.draw_landmarks(
                        img,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS
                    )
            # FPS during calibration
            c_time = time.time()
            fps    = int(1 / (c_time - p_time + 1e-9))
            p_time = c_time
            cv2.putText(img, f"FPS: {fps}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.imshow("Magic Hand — Virtual Mouse", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            continue
        else:
            # Finish calibration
            if len(calib_spans) >= 5:
                baseline       = float(np.median(calib_spans))
                adaptive_pinch  = baseline * PINCH_RATIO
                adaptive_scroll = baseline * SCROLL_RATIO
            # else keep defaults
            calibrating = False

    # ── PER-FRAME PROCESSING ───────────────────────────
    confidence_ok = False

    if results.multi_hand_landmarks and results.multi_handedness:
        conf          = results.multi_handedness[0].classification[0].score
        confidence_ok = (conf >= MIN_HAND_CONFIDENCE)

        # Confidence bar (top-right)
        bar_x = w - 145
        cv2.putText(img, "Confidence", (bar_x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
        bar_color = (0,200,80) if confidence_ok else (0,80,200)
        draw_progress_bar(img, bar_x, 25, 120, 8, conf, 1.0, bar_color)
        cv2.putText(img, f"{conf:.2f}", (bar_x+124, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

        if not confidence_ok:
            overlay_text(img, "Low confidence — adjust lighting / distance",
                         (FR+10, h-30), scale=0.6, color=(0,80,255))
        else:
            handLms = results.multi_hand_landmarks[0]
            lm = [(int(lk.x*w), int(lk.y*h)) for lk in handLms.landmark]
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            index_tip  = lm[8]
            thumb_tip  = lm[4]
            middle_tip = lm[12]

            # ── Update adaptive thresholds ─────────────
            span = hand_span(lm)
            hand_span_history.append(span)
            if len(hand_span_history) >= 10:
                ref_span        = float(np.median(hand_span_history))
                adaptive_pinch  = ref_span * PINCH_RATIO
                adaptive_scroll = ref_span * SCROLL_RATIO

            PT = adaptive_pinch
            ST = adaptive_scroll

            # ── Cursor mapping ─────────────────────────
            ix = np.clip(index_tip[0], FR, w-FR)
            iy = np.clip(index_tip[1], FR, h-FR)
            sx = np.interp(ix, (FR, w-FR), (0, screen_w))
            sy = np.interp(iy, (FR, h-FR), (0, screen_h))
            cx = prev_x + (sx - prev_x) / SMOOTHENING
            cy = prev_y + (sy - prev_y) / SMOOTHENING
            pyautogui.moveTo(int(cx), int(cy))
            prev_x, prev_y = cx, cy

            cv2.circle(img, index_tip, 12, (255, 0, 255), cv2.FILLED)

            # ── Distances ─────────────────────────────
            d_ti = dist(thumb_tip,  index_tip)
            d_tm = dist(thumb_tip,  middle_tip)
            d_im = dist(index_tip,  middle_tip)

            r_ti = d_ti / max(span, 1)
            r_tm = d_tm / max(span, 1)
            r_im = d_im / max(span, 1)

            # ── Pinch hold counters ────────────────────
            ti_frames = ti_frames+1 if d_ti < PT else 0
            tm_frames = tm_frames+1 if d_tm < PT else 0
            im_frames = im_frames+1 if d_im < PT else 0

            current_time = time.time()
            can_click    = (current_time - last_click_time) > CLICK_DELAY

            # ── Colour-coded pinch indicators ─────────
            draw_indicator(img, thumb_tip,  index_tip,  r_ti, PINCH_RATIO)
            draw_indicator(img, thumb_tip,  middle_tip, r_tm, PINCH_RATIO)
            draw_indicator(img, index_tip,  middle_tip, r_im, PINCH_RATIO * 1.5)

            # ── Hold progress bars (bottom-left HUD) ──
            hud_x, hud_y = 20, h - 115
            labels  = ["L", "D", "R"]
            frames  = [ti_frames, tm_frames, im_frames]
            colours = [(0,220,80), (0,180,255), (0,80,255)]
            for i, (label, frm, col) in enumerate(zip(labels, frames, colours)):
                yy = hud_y + i*18
                cv2.putText(img, label, (hud_x, yy+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
                draw_progress_bar(img, hud_x+16, yy+2, 60, 8,
                                  frm, HOLD_FRAMES, col)

            # ── Gesture priority ladder ────────────────
            any_pinch = (ti_frames >= HOLD_FRAMES or
                         tm_frames >= HOLD_FRAMES or
                         im_frames >= HOLD_FRAMES)

            if ti_frames >= HOLD_FRAMES and can_click:
                pyautogui.click()
                gesture_text    = "LEFT CLICK"
                last_click_time = current_time
                ti_frames       = 0

            elif tm_frames >= HOLD_FRAMES and can_click:
                pyautogui.doubleClick()
                gesture_text    = "DOUBLE CLICK"
                last_click_time = current_time
                tm_frames       = 0

            elif im_frames >= HOLD_FRAMES and can_click:
                pyautogui.rightClick()
                gesture_text    = "RIGHT CLICK"
                last_click_time = current_time
                im_frames       = 0

            elif not any_pinch:
                # ── Momentum scroll ────────────────────
                if d_im > ST:
                    scroll_hold += 1
                else:
                    scroll_hold   = 0
                    last_scroll_y = None

                if scroll_hold >= SCROLL_HOLD_FRAMES:
                    if last_scroll_y is not None:
                        delta = last_scroll_y - index_tip[1]
                        if abs(delta) > 4:          # dead-zone
                            scroll_buffer.append(delta)
                            avg   = float(np.mean(scroll_buffer))
                            amt   = int(np.sign(avg) *
                                        max(1, abs(avg) / 12))
                            pyautogui.scroll(amt * SCROLL_SPEED)
                            gesture_text = ("SCROLL UP"
                                            if avg > 0
                                            else "SCROLL DOWN")
                    last_scroll_y = index_tip[1]

            # ── Adaptive threshold status (tiny) ──────
            cv2.putText(img,
                        f"pinch:{PT:.0f}px  scroll:{ST:.0f}px  span:{span:.0f}px",
                        (FR+5, h-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,100), 1)

    # ── Gesture label ──────────────────────────────────
    if gesture_text:
        overlay_text(img, gesture_text, (20, 70),
                     scale=1.0, color=(0,255,0), thickness=2)

    # ── FPS ───────────────────────────────────────────
    c_time = time.time()
    fps    = int(1 / (c_time - p_time + 1e-9))
    p_time = c_time
    cv2.putText(img, f"FPS: {fps}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    # ── Cheat sheet ───────────────────────────────────
    hints = [
        "Thumb+Index   = Left click",
        "Thumb+Middle  = Double click",
        "Index+Middle  = Right click",
        "Spread fingers = Scroll",
        "C = Recalibrate",
    ]
    for i, hint in enumerate(hints):
        cv2.putText(img, hint, (w-300, h-100+i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    cv2.imshow("Magic Hand — Virtual Mouse", img)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord('c'):
        calibrating  = True
        calib_start  = time.time()
        calib_spans  = []

cap.release()
cv2.destroyAllWindows()