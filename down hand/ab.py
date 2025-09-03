import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

triggered = False
trigger_delay = 1  # seconds
last_trigger_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = img.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Wrist landmark
            wrist = handLms.landmark[0]
            wrist_y = wrist.y * h

            # Agar wrist screen ke neeche 2/3 hissa cross kar le
            if wrist_y > (h * 0.66):
                current_time = time.time()
                if not triggered or (current_time - last_trigger_time > trigger_delay):
                    pyautogui.hotkey('win', 'd')  # Desktop show
                    triggered = True
                    last_trigger_time = current_time
            else:
                triggered = False

    cv2.imshow("Hand Down to Desktop", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()