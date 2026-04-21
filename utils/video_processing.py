import cv2
import numpy as np
import matplotlib.pyplot as plt

def track_pendulum(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Falls OpenCV die FPS nicht lesen kann, setzen wir einen Standardwert (z.B. 30)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0 
        print("Warnung: FPS konnte nicht gelesen werden, nutze Standard: 30")

    positions = []
    times = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # HSV-Filter
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 80, 80]) # Etwas großzügigerer Bereich
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            # Nur tracken, wenn das Objekt groß genug ist (Rauschunterdrückung)
            if cv2.contourArea(largest) > 100: 
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    
                    # Hier speichern wir Zeit und Position
                    current_time = frame_idx / fps
                    positions.append(cx)
                    times.append(current_time)
                    
                    # VISUALISIERUNG: Zeichne einen Punkt auf das Pendel (optional)
                    cv2.circle(frame, (cx, int(M["m01"]/M["m00"])), 10, (0, 255, 0), -1)

        # Zeige das Video live an, damit du siehst, ob die Maske funktioniert
        cv2.imshow('Tracking-Check', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' drücken zum Abbrechen
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    
    # Plot erstellen
    plt.plot(times, positions)
    plt.xlabel('Zeit (s)')
    plt.ylabel('X-Position (Pixel)')
    plt.title('Pendelschwingung')
    plt.show()

# Aufruf der Funktion
track_pendulum('dein_video.mp4')
