import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
from inference.predictor import predict_video
from training.gradcam import generate_gradcam
from utils.report_generator import generate_report
from inference.multimodal_predictor import multimodal_predict

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 50%,
        #111827 100%
    );
}

/* Main text */
html, body, [class*="css"] {
    color: white;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #38bdf8;
    border-radius: 15px;
    padding: 15px;
    background-color: rgba(255,255,255,0.03);
}

/* Metrics cards */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Images */
img {
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

st.title("🎭 DeepFake Detection System")
st.caption("AI-powered video forgery detection using deep learning and explainable AI")

st.sidebar.title("DeepFake Detection")
st.sidebar.info(   """
    Upload a video and the AI will analyze it for signs of manipulation.

    Features:\n
    • Video Analysis\n
    • Confidence Score\n
    • Risk Assessment\n
    • GradCAM Visualization\n
    • PDF Report
    """
)

st.write("Upload a video and the AI will determine whether it is REAL or Fake.")

uploaded_file = st.file_uploader("Choose a video", type = ["mp4", "avi", "mov"])
if uploaded_file is not None:
    st.write(f"FileName: {uploaded_file.name}")
    st.write(f"Size: {uploaded_file.size / 1024:.2f} KB")
    st.video(uploaded_file)

if uploaded_file is not None:
    if st.button("Analyze Video"):
        with st.spinner("Analyzing video..."):

            temp_video_path = uploaded_file.name

            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            prediction, confidence, video_conf, audio_conf = multimodal_predict(temp_video_path)

        label = prediction
        st.subheader("Analysis Result")

        if label == "REAL":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Confidence Score:", f"{confidence:.2f}%")
        with col2:
            st.metric("Prediction:", label)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Video Confidence",
                f"{video_conf:.2f}%"
            )

        with col2:
            st.metric(
                "Audio Confidence",
                f"{audio_conf:.2f}%"
            )

        with col3:
            st.metric(
                "Fusion Score",
                f"{confidence:.2f}%"
            )

        if prediction == "FAKE":
            if confidence >= 80:
                risk = "HIGH RISK"
            elif confidence >= 50:
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
        pdf_path = generate_report(label, confidence, risk, video_conf, audio_conf, gradcam_path)
        
        st.subheader("AI ANALYSIS")
        st.image(gradcam_path, caption = "Grad-CAM Visualization")

        with open(pdf_path, "rb") as file:
            st.download_button(label = "Download Analysis Report", data = file,
                               file_name = "deepfake_report.pdf", mime = "application/pdf")
            
st.markdown("---")
st.caption("Developed by Rohan Dudeja | DeepFake Detection System | PyTorch + Streamlit + GradCAM")