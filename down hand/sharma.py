import cv2
import mediapipe as mp
import pyautogui
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.85,
                      min_tracking_confidence=0.85)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

gesture_time = time.time()
gesture_cooldown = 1.5  # seconds

prev_x = None  # previous hand X position

def fingers_up(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    thumb_ip = 3
    fingers = []

    # Thumb
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip_id in finger_tips:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            finger_count = fingers_up(handLms)
            wrist_x = handLms.landmark[0].x * w
            now = time.time()

            # 1. Fist detection -> Close all windows (Alt + F4 repeatedly)
            if finger_count == 0 and (now - gesture_time > gesture_cooldown):
                pyautogui.hotkey('alt', 'f4')
                cv2.putText(frame, 'Closing Window', (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                gesture_time = now

            # 2. Open palm -> Switch virtual desktops
            elif finger_count == 5:
                if prev_x is not None:
                    movement = wrist_x - prev_x
                    if abs(movement) > 60 and (now - gesture_time > gesture_cooldown):
                        if movement > 0:
                            pyautogui.hotkey('ctrl', 'win', 'right')
                            cv2.putText(frame, 'Next Desktop', (200, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        else:
                            pyautogui.hotkey('ctrl', 'win', 'left')
                            cv2.putText(frame, 'Previous Desktop', (200, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        gesture_time = now
                prev_x = wrist_x
            else:
                prev_x = None

    cv2.imshow("Virtual Desktop Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
