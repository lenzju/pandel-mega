import tensorflow as tf
import cv2
import numpy as np

def classify_states(video_path):
    try:
        model = tf.keras.models.load_model("models/keras_model.h5")
    except:
        return ["Kein Modell gefunden"]

    cap = cv2.VideoCapture(video_path)
    states = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (224, 224))
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        label = np.argmax(prediction)

        states.append(label)

    cap.release()
    return states
