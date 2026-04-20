import cv2
import numpy as np

def track_pendulum(video_path):
    cap = cv2.VideoCapture(video_path)

    positions = []
    times = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Gelb-Filter
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                positions.append(cx)
                times.append(frame_idx / fps)

        frame_idx += 1

    cap.release()

    return np.array(positions), np.array(times)
