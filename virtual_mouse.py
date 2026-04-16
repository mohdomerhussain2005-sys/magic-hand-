import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe setup (optimized)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0  # FAST mode
)
mp_draw = mp.solutions.drawing_utils

# Camera setup (HD + FPS boost)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Settings
prev_x, prev_y = 0, 0
smoothening = 4   # Higher = smoother
click_delay = 0.4
last_click_time = 0
frame_reduction = 30

# FPS calculation
p_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    # Draw control area
    cv2.rectangle(img, (frame_reduction, frame_reduction),
                  (w - frame_reduction, h - frame_reduction),
                  (255, 0, 255), 2)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture_text = "None"

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]

        lm_list = []
        for id, lm in enumerate(handLms.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))

        mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        # Fingertips
        x1, y1 = lm_list[8]   # Index
        x2, y2 = lm_list[4]   # Thumb
        x3, y3 = lm_list[12]  # Middle

        # Draw pointer
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

        # Convert coordinates
        screen_x = np.interp(x1, (frame_reduction, w - frame_reduction), (0, screen_w))
        screen_y = np.interp(y1, (frame_reduction, h - frame_reduction), (0, screen_h))

        # Smooth movement
        curr_x = prev_x + (screen_x - prev_x) / smoothening
        curr_y = prev_y + (screen_y - prev_y) / smoothening

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Distances
        dist_thumb_index = np.hypot(x2 - x1, y2 - y1)
        dist_index_middle = np.hypot(x3 - x1, y3 - y1)
        dist_thumb_middle = np.hypot(x2 - x3, y2 - y3)

        current_time = time.time()

        # LEFT CLICK
        if dist_thumb_index < 25 and (current_time - last_click_time) > click_delay:
            pyautogui.click()
            gesture_text = "LEFT CLICK"
            last_click_time = current_time

        # RIGHT CLICK
        elif dist_index_middle < 25 and (current_time - last_click_time) > click_delay:
            pyautogui.rightClick()
            gesture_text = "RIGHT CLICK"
            last_click_time = current_time

        # DOUBLE CLICK
        elif dist_thumb_middle < 25 and (current_time - last_click_time) > click_delay:
            pyautogui.doubleClick()
            gesture_text = "DOUBLE CLICK"
            last_click_time = current_time

        # SCROLL
        elif dist_index_middle > 40:
            if y1 < y3:
                pyautogui.scroll(40)
                gesture_text = "SCROLL UP"
            else:
                pyautogui.scroll(-40)
                gesture_text = "SCROLL DOWN"

        # ZOOM
        if dist_thumb_index < 40:
            pyautogui.keyDown('ctrl')
            pyautogui.scroll(30)
            pyautogui.keyUp('ctrl')
            gesture_text = "ZOOM"

        # Show gesture text
        cv2.putText(img, gesture_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 3)

    # FPS display
    c_time = time.time()
    fps = int(1 / (c_time - p_time)) if (c_time - p_time) != 0 else 0
    p_time = c_time

    cv2.putText(img, f'FPS: {fps}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2)

    cv2.imshow("Virtual Mouse PRO", img)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()