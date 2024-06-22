import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Camera properties
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

def set_volume(volume):
    # Construct the osascript command to set the volume
    command = f"osascript -e 'set volume output volume {volume}'"
    subprocess.call(command, shell=True)

with mp_hands.Hands(
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2) as hands:

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("Failed to read from camera.")
            break

        # Process image with MediaPipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract landmarks and calculate volume based on hand position
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                if len(lmList) != 0:
                    x1, y1 = lmList[4][1], lmList[4][2]
                    x2, y2 = lmList[8][1], lmList[8][2]

                    # Calculate length between fingertips
                    length = math.hypot(x2 - x1, y2 - y1)

                    # Interpolate volume based on fingertip distance
                    volume_percent = np.interp(length, [50, 220], [0, 100])
                    set_volume(volume_percent)

                    # Draw visual feedback (volume bar)
                    cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                    volBar = np.interp(length, [50, 220], [400, 150])
                    cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'{int(volume_percent)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                                1, (0, 0, 0), 3)

        # Display the image with landmarks and volume info
        cv2.imshow('Hand Volume Control', image)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
