import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Camera
cap = cv2.VideoCapture(0)

# Drawing canvas
canvas = None

prev_x, prev_y = None, None

# Drawing color
draw_color = (255, 0, 255)   # Purple


while True:

    success, frame = cap.read()

    if not success:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = frame.shape

            # Index finger
            index = hand_landmarks.landmark[8]

            # Thumb
            thumb = hand_landmarks.landmark[4]

            x = int(index.x * w)
            y = int(index.y * h)

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            # Distance between thumb and index
            distance = np.sqrt(
                (x - thumb_x) ** 2 +
                (y - thumb_y) ** 2
            )

            # 🤏 Eraser
            if distance < 40:

                cv2.circle(
                    canvas,
                    (x, y),
                    30,
                    (0, 0, 0),
                    -1
                )

                prev_x, prev_y = None, None

                cv2.putText(
                    frame,
                    "ERASER",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # ☝️ Drawing
            else:

                cv2.circle(
                    frame,
                    (x, y),
                    10,
                    draw_color,
                    -1
                )

                if prev_x is not None and prev_y is not None:

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        draw_color,
                        8
                    )

                prev_x, prev_y = x, y

            # Hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    else:
        prev_x, prev_y = None, None

    # Combine camera and drawing
    output = cv2.add(frame, canvas)

    cv2.imshow("Air Drawing App", output)

    # Keyboard
    key = cv2.waitKey(1) & 0xFF

    # Purple
    if key == ord("p"):
        draw_color = (255, 0, 255)

    # Blue
    if key == ord("b"):
        draw_color = (255, 0, 0)

    # Green
    if key == ord("g"):
        draw_color = (0, 255, 0)

    # Clear
    if key == ord("c"):
        canvas[:] = 0
        prev_x, prev_y = None, None
        # Save drawing
    if key == ord("s"):
     cv2.imwrite("my_drawing.png", canvas)
    print("Drawing saved as my_drawing.png")

    # Quit
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()