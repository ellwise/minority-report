import math
import time
from threading import Thread

import cv2
import mediapipe as mp
import pyautogui
from typer import Typer

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

app = Typer()


@app.command()
def _():
    """Run the application"""

    xsz, ysz = pyautogui.size()

    shared_data = {
        "thumb_tip": None,
        "left_click": False,
    }

    def track_hands(shared_data):

        context_manager = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5,
        )

        with context_manager as hands:
            cap = cv2.VideoCapture(cv2.CAP_V4L2)
            # use threads...
            while cap.isOpened():
                _, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                    thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
                    left_click = 0
                    right_click = 0
                    norm = 0
                    for attr in ("x", "y", "z"):
                        left_vec = getattr(thumb_tip, attr) - getattr(index_finger_tip, attr)
                        left_click += left_vec**2
                        right_vec = getattr(thumb_tip, attr) - getattr(middle_finger_tip, attr)
                        right_click += right_vec**2
                        norm_vec = getattr(thumb_tip, attr) - getattr(wrist, attr)
                        norm += norm_vec**2
                    norm = math.sqrt(norm)
                    left_click = math.sqrt(left_click / norm)
                    right_click = math.sqrt(right_click / norm)

                    shared_data["thumb_tip"] = thumb_tip
                    shared_data["left_click"] = left_click

                cv2.imshow("Hand tracking", image)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def move_mouse(shared_data):
        while True:
            time.sleep(1 / 30)
            try:
                thumb_tip = shared_data["thumb_tip"]
                left_click = shared_data["left_click"]
                # [.4, .6] -> [0, xsz]
                # x -> 5 * (x - 0.4) * xsz
                target_x = 5 * (thumb_tip.x - 0.4) * xsz
                target_y = 5 * (thumb_tip.y - 0.4) * ysz
                transition_time = 1 / 30
                pyautogui.moveTo(target_x, target_y, transition_time)

                if left_click < 0.1:
                    print("LEFT CLICK")
                    pyautogui.click()
                else:
                    print("NO CLICK")
            except Exception:
                pass

    thread_1 = Thread(target=track_hands, args=(shared_data,))
    thread_2 = Thread(target=move_mouse, args=(shared_data,), daemon=True)
    thread_1.start()
    thread_2.start()
