import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

# ══════════════════════════════════════════════════════
#  MAGIC HAND — Virtual Mouse  (v4)
#  Clean: no connector lines, bold gesture cards,
#         distinct gesture zones, momentum scroll
# ══════════════════════════════════════════════════════

SMOOTHENING         = 2
FRAME_REDUCTION     = 80
CLICK_DELAY         = 0.4
HOLD_FRAMES         = 4       # slightly longer hold = fewer accidents
SCROLL_HOLD_FRAMES  = 2
SCROLL_SPEED        = 4
CALIBRATION_SECS    = 3
MIN_HAND_CONFIDENCE = 0.75
ADAPT_WINDOW        = 60
PINCH_RATIO         = 0.13    # thumb+index  — left click
DOUBLE_RATIO        = 0.13    # thumb+middle — double click (same dist, diff fingers)
RIGHT_RATIO         = 0.13    # index+middle — right click
SCROLL_RATIO        = 0.22    # index+middle FAR apart — scroll

# ── Gesture colours (BGR) ──────────────────────────────
COL = {
    "LEFT CLICK":   (0,  220, 80),
    "DOUBLE CLICK": (0,  180, 255),
    "RIGHT CLICK":  (0,  80,  255),
    "SCROLL UP":    (255,200, 0),
    "SCROLL DOWN":  (255,140, 0),
}

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
draw_spec_gray = mp_draw.DrawingSpec(color=(80,80,80), thickness=1, circle_radius=2)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# ── State ──────────────────────────────────────────────
prev_x, prev_y    = 0, 0
last_click_time   = 0
p_time            = 0
ti_frames = tm_frames = im_frames = 0
scroll_buffer     = deque(maxlen=5)
last_scroll_y     = None
scroll_hold       = 0
hand_span_history = deque(maxlen=ADAPT_WINDOW)
adaptive_pinch    = 35.0
adaptive_scroll   = 60.0
calibrating       = True
calib_start       = time.time()
calib_spans       = []

# Gesture flash state (makes the card linger briefly after firing)
flash_text        = ""
flash_timer       = 0.0
FLASH_DURATION    = 0.5          # seconds the card stays on screen

# ── Helpers ────────────────────────────────────────────
def dist(p1, p2):
    return float(np.hypot(p1[0]-p2[0], p1[1]-p2[1]))

def hand_span(lm):
    return dist(lm[0], lm[9])

def draw_progress_bar(img, x, y, bw, bh, value, max_val, color):
    cv2.rectangle(img, (x, y), (x+bw, y+bh), (40,40,40), -1)
    filled = int(bw * min(value / max(max_val, 1), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x+filled, y+bh), color, -1)

def draw_gesture_card(img, text):
    """Big centred card that flashes when a gesture fires."""
    if not text:
        return
    color  = COL.get(text, (255,255,255))
    font   = cv2.FONT_HERSHEY_DUPLEX
    scale  = 1.4
    thick  = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    img_h, img_w = img.shape[:2]
    pad    = 18
    rx     = (img_w - tw) // 2 - pad
    ry     = img_h - 120
    rw     = tw + pad*2
    rh     = th + pad*2

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Coloured border
    cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), color, 2)

    # Text
    tx = rx + pad
    ty = ry + pad + th
    cv2.putText(img, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

def draw_fingertip_dot(img, pt, color, radius=14):
    """Solid dot with a thin white ring — clean and easy to see."""
    cv2.circle(img, pt, radius,     color,       cv2.FILLED)
    cv2.circle(img, pt, radius + 2, (240,240,240), 1)

def draw_hold_arc(img, center, frames, max_frames, color):
    """
    Circular arc around a fingertip that fills up as you hold the pinch.
    Replaces the old connector lines — shows progress without clutter.
    """
    if frames <= 0:
        return
    t      = min(frames / max_frames, 1.0)
    angle  = int(360 * t)
    radius = 22
    # Background ring
    cv2.ellipse(img, center, (radius,radius), -90, 0, 360,
                (60,60,60), 2, cv2.LINE_AA)
    # Filled arc
    cv2.ellipse(img, center, (radius,radius), -90, 0, angle,
                color, 3, cv2.LINE_AA)

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

    cv2.rectangle(img, (FR, FR), (w-FR, h-FR), (180, 0, 180), 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture_text = ""

    # ── CALIBRATION ───────────────────────────────────
    if calibrating:
        elapsed   = time.time() - calib_start
        remaining = CALIBRATION_SECS - elapsed
        if remaining > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,
                        f"Open hand to calibrate  {remaining:.1f}s",
                        (FR+10, h//2 - 16), font, 0.9, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, "Press C to recalibrate anytime",
                        (FR+10, h//2 + 16), font, 0.55, (160,160,160), 1)
            if results.multi_hand_landmarks and results.multi_handedness:
                conf = results.multi_handedness[0].classification[0].score
                if conf >= MIN_HAND_CONFIDENCE:
                    lm = [(int(lk.x*w), int(lk.y*h))
                          for lk in results.multi_hand_landmarks[0].landmark]
                    calib_spans.append(hand_span(lm))
                    mp_draw.draw_landmarks(
                        img,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS,
                        draw_spec_gray, draw_spec_gray
                    )
            c_time = time.time()
            fps    = int(1 / (c_time - p_time + 1e-9))
            p_time = c_time
            cv2.putText(img, f"FPS: {fps}", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.imshow("Magic Hand — Virtual Mouse", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            continue
        else:
            if len(calib_spans) >= 5:
                baseline        = float(np.median(calib_spans))
                adaptive_pinch  = baseline * PINCH_RATIO
                adaptive_scroll = baseline * SCROLL_RATIO
            calibrating = False

    # ── PER-FRAME ──────────────────────────────────────
    if results.multi_hand_landmarks and results.multi_handedness:
        conf = results.multi_handedness[0].classification[0].score

        # Confidence bar (top right, compact)
        bar_x = w - 140
        cv2.putText(img, f"conf {conf:.2f}", (bar_x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140,140,140), 1)
        bar_col = (0,180,60) if conf >= MIN_HAND_CONFIDENCE else (0,60,200)
        draw_progress_bar(img, bar_x, 22, 110, 6, conf, 1.0, bar_col)

        if conf < MIN_HAND_CONFIDENCE:
            cv2.putText(img, "Low confidence — adjust lighting",
                        (FR+10, h-30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0,60,220), 1)
        else:
            handLms = results.multi_hand_landmarks[0]
            lm = [(int(lk.x*w), int(lk.y*h)) for lk in handLms.landmark]

            # Draw skeleton in gray (quiet, not distracting)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS,
                                   draw_spec_gray, draw_spec_gray)

            index_tip  = lm[8]
            thumb_tip  = lm[4]
            middle_tip = lm[12]
            ring_tip   = lm[16]

            # Update adaptive thresholds
            span = hand_span(lm)
            hand_span_history.append(span)
            if len(hand_span_history) >= 10:
                ref             = float(np.median(hand_span_history))
                adaptive_pinch  = ref * PINCH_RATIO
                adaptive_scroll = ref * SCROLL_RATIO
            PT = adaptive_pinch
            ST = adaptive_scroll

            # Cursor movement — driven by index fingertip only
            ix = np.clip(index_tip[0], FR, w-FR)
            iy = np.clip(index_tip[1], FR, h-FR)
            sx = np.interp(ix, (FR, w-FR), (0, screen_w))
            sy = np.interp(iy, (FR, h-FR), (0, screen_h))
            cx = prev_x + (sx - prev_x) / SMOOTHENING
            cy = prev_y + (sy - prev_y) / SMOOTHENING
            pyautogui.moveTo(int(cx), int(cy))
            prev_x, prev_y = cx, cy

            # Distances
            d_ti = dist(thumb_tip,  index_tip)    # LEFT CLICK
            d_tm = dist(thumb_tip,  middle_tip)   # DOUBLE CLICK
            d_im = dist(index_tip,  middle_tip)   # RIGHT CLICK / SCROLL

            # ── Pinch hold counters ────────────────────
            ti_frames = ti_frames + 1 if d_ti < PT else 0
            tm_frames = tm_frames + 1 if d_tm < PT else 0
            im_frames = im_frames + 1 if d_im < PT else 0

            current_time = time.time()
            can_click    = (current_time - last_click_time) > CLICK_DELAY
            any_pinch    = (ti_frames >= HOLD_FRAMES or
                            tm_frames >= HOLD_FRAMES or
                            im_frames >= HOLD_FRAMES)

            # ── Fingertip dots — colour shows which finger is active ──
            # Index fingertip — always the pointer (magenta)
            draw_fingertip_dot(img, index_tip, (220, 0, 220))

            # Thumb — lights up green when approaching index (left click ready)
            thumb_col = (0,220,80) if ti_frames > 0 else (160,160,160)
            draw_fingertip_dot(img, thumb_tip, thumb_col, radius=10)

            # Middle — lights up cyan when approaching thumb (double click)
            #          lights up orange when approaching index (right click)
            if tm_frames > 0:
                mid_col = (0, 180, 255)   # cyan = double click
            elif im_frames > 0:
                mid_col = (0,  80, 255)   # orange-red = right click
            else:
                mid_col = (160,160,160)
            draw_fingertip_dot(img, middle_tip, mid_col, radius=10)

            # ── Hold arcs (progress ring around the active fingertip) ──
            # Only draw arc for the gesture that's building up — no clutter
            if ti_frames > 0 and tm_frames == 0 and im_frames == 0:
                draw_hold_arc(img, index_tip,  ti_frames, HOLD_FRAMES, (0,220,80))
                draw_hold_arc(img, thumb_tip,  ti_frames, HOLD_FRAMES, (0,220,80))

            elif tm_frames > 0 and ti_frames == 0:
                draw_hold_arc(img, thumb_tip,  tm_frames, HOLD_FRAMES, (0,180,255))
                draw_hold_arc(img, middle_tip, tm_frames, HOLD_FRAMES, (0,180,255))

            elif im_frames > 0 and ti_frames == 0:
                draw_hold_arc(img, index_tip,  im_frames, HOLD_FRAMES, (0,80,255))
                draw_hold_arc(img, middle_tip, im_frames, HOLD_FRAMES, (0,80,255))

            # ── Gesture ladder ─────────────────────────
            if ti_frames >= HOLD_FRAMES and can_click:
                pyautogui.click()
                flash_text      = "LEFT CLICK"
                flash_timer     = time.time()
                last_click_time = current_time
                ti_frames       = 0

            elif tm_frames >= HOLD_FRAMES and can_click:
                pyautogui.doubleClick()
                flash_text      = "DOUBLE CLICK"
                flash_timer     = time.time()
                last_click_time = current_time
                tm_frames       = 0

            elif im_frames >= HOLD_FRAMES and can_click:
                pyautogui.rightClick()
                flash_text      = "RIGHT CLICK"
                flash_timer     = time.time()
                last_click_time = current_time
                im_frames       = 0

            elif not any_pinch:
                # Momentum scroll — index+middle spread wide
                if d_im > ST:
                    scroll_hold += 1
                else:
                    scroll_hold   = 0
                    last_scroll_y = None

                if scroll_hold >= SCROLL_HOLD_FRAMES:
                    if last_scroll_y is not None:
                        delta = last_scroll_y - index_tip[1]
                        if abs(delta) > 4:
                            scroll_buffer.append(delta)
                            avg = float(np.mean(scroll_buffer))
                            amt = int(np.sign(avg) * max(1, abs(avg)/12))
                            pyautogui.scroll(amt * SCROLL_SPEED)
                            gesture_text = ("SCROLL UP"
                                            if avg > 0 else "SCROLL DOWN")
                    last_scroll_y = index_tip[1]

            # Tiny adaptive status (bottom edge)
            cv2.putText(img,
                        f"pinch:{PT:.0f}  scroll:{ST:.0f}  span:{span:.0f}",
                        (FR+4, h-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (90,90,90), 1)

    # ── Gesture flash card ────────────────────────────
    # Show scroll label immediately; show click label for FLASH_DURATION after firing
    display_text = gesture_text  # live scroll text
    if not display_text and flash_text:
        if time.time() - flash_timer < FLASH_DURATION:
            display_text = flash_text
        else:
            flash_text = ""

    if display_text:
        draw_gesture_card(img, display_text)

    # ── FPS ───────────────────────────────────────────
    c_time = time.time()
    fps    = int(1 / (c_time - p_time + 1e-9))
    p_time = c_time
    cv2.putText(img, f"FPS:{fps}", (20,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,0), 2)

    # ── Cheat sheet (bottom-right, minimal) ───────────
    hints = [
        "Thumb + Index  tip = Left click",
        "Thumb + Middle tip = Double click",
        "Index + Middle tip = Right click",
        "Spread Index+Middle = Scroll",
        "C = recalibrate",
    ]
    for i, hint in enumerate(hints):
        cv2.putText(img, hint,
                    (w - 330, h - 106 + i*21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1)

    cv2.imshow("Magic Hand — Virtual Mouse", img)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    if key == ord('c'):
        calibrating = True
        calib_start = time.time()
        calib_spans = []

cap.release()
cv2.destroyAllWindows()
