import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from inference.predictor import predict_video
from training.gradcam import generate_gradcam
from utils.report_generator import generate_report

st.title("🎭 DeepFake Detection System")

st.write("Upload a video and the AI will determine whether it is REAL or Fake.")

uploaded_file = st.file_uploader("Choose a video", type = ["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

if uploaded_file is not None:
    if st.button("Analyze Video"):
        with st.spinner("Analyzing video..."):

            temp_video_path = uploaded_file.name

            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            prediction, confidence = predict_video(temp_video_path)

        if prediction == 0:
            label = "REAL"
        else:
            label = "FAKE"

        st.subheader("Analysis Result")

        if label == "REAL":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.metric("Confidence", f"{confidence:.2f}%")

        if confidence >= 90:
            risk = "HIGH RISK"
        elif confidence >= 70:
            risk = "MEDIUM RISK"
        else:
            risk = "LOW RISK"

        if risk == "HIGH RISK":
            st.error(f"Risk Level: {risk}")
        elif risk == "MEDIUM RISK":
            st.warning(f"Risk Level: {risk}")
        else:
            st.success(f"Risk Level: {risk}")

        gradcam_path = generate_gradcam(temp_video_path)
        pdf_path = generate_report(label, confidence, risk)
        st.image(gradcam_path, caption = "Grad-CAM Visualization")

        with open(pdf_path, "rb") as file:
            st.download_button(label = "Download Analysis Report", data = file,
                               file_name = "deepfake_report.pdf", mime = "application/pdf")