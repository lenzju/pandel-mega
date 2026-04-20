import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import cv2

from utils.video_processing import track_pendulum
from utils.physics import analyze_motion
from utils.ml_model import classify_states

st.title("🎾 Pendel-Analyse App")

st.write("Lade ein Video eines Pendels (gelber Tennisball) hoch.")

video_file = st.file_uploader("MP4 Video hochladen", type=["mp4"])

length = st.number_input("Pendellänge (in Metern)", min_value=0.1, value=1.0)

use_ml = st.checkbox("ML-Modell (Teachable Machine) verwenden")

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    st.video(tfile.name)

    st.write("🔍 Tracking läuft...")

    positions, times = track_pendulum(tfile.name)

    st.success("Tracking abgeschlossen!")

    # Physik-Auswertung
    period, frequency, g = analyze_motion(times, positions, length)

    st.subheader("📊 Ergebnisse")
    st.write(f"Periodendauer: {period:.2f} s")
    st.write(f"Frequenz: {frequency:.2f} Hz")
    st.write(f"Erdbeschleunigung: {g:.2f} m/s²")

    # Plot
    st.subheader("📈 Position-Zeit-Diagramm")

    fig, ax = plt.subplots()
    ax.plot(times, positions)
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Position (Pixel)")
    ax.set_title("Pendelbewegung")

    st.pyplot(fig)

    # ML Vergleich
    if use_ml:
        st.subheader("🤖 ML-Modell Analyse")

        states = classify_states(tfile.name)

        st.write("Zustände erkannt:", states[:20])
