import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)
clicking = False  # Prevent multiple clicks

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror image
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions (pixel coordinates)
            index_finger = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_finger = (int(lm[4].x * w), int(lm[4].y * h))
            middle_finger = (int(lm[12].x * w), int(lm[12].y * h))
            ring_finger = (int(lm[16].x * w), int(lm[16].y * h))
            pinky_finger = (int(lm[20].x * w), int(lm[20].y * h))

            # Draw index finger pointer
            cv2.circle(frame, index_finger, 10, (255, 0, 255), cv2.FILLED)

            # Map camera position to screen
            screen_x = np.interp(index_finger[0], [0, w], [0, screen_w])
            screen_y = np.interp(index_finger[1], [0, h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y, duration=0.01)

            # LEFT CLICK: Thumb + Index
            if distance(index_finger, thumb_finger) < 40:
                if not clicking:
                    pyautogui.click()
                    clicking = True
                    cv2.circle(frame, index_finger, 15, (0, 255, 0), cv2.FILLED)

            # RIGHT CLICK: Thumb + Pinky
            elif distance(pinky_finger, thumb_finger) < 40:
                if not clicking:
                    pyautogui.rightClick()
                    clicking = True
                    cv2.putText(frame, "Right Click", (10, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                clicking = False

            # SCROLL UP: Thumb + Middle
            if distance(middle_finger, thumb_finger) < 40:
                pyautogui.scroll(30)
                cv2.putText(frame, "Scroll Up", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # SCROLL DOWN: Thumb + Ring
            if distance(ring_finger, thumb_finger) < 40:
                pyautogui.scroll(-30)
                cv2.putText(frame, "Scroll Down", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show webcam feed
    cv2.imshow("Hand Gesture Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
