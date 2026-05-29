import streamlit as st

st.title("DeepFake Detection System")

st.write("Upload a video and the AI will determine whether it is REAL or Fake.")

uploaded_file = st.file_uploader("Choose a video", type = ["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

#if uploaded_file is not None:
   # if st.button("Analyze Video"):
    #    s