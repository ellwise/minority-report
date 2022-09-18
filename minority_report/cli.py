import cv2
from typer import Typer

# import mediapipe as mp

app = Typer()


@app.command()
def _():
    """Run the application"""

    # mp_drawing = mp.solutions.drawing_utils
    # mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(cv2.CAP_V4L2)
    while cap.isOpened():
        _, frame = cap.read()
        cv2.imshow("Hand tracking", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
