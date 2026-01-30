import streamlit as st
import tempfile
from pathlib import Path
from src.app_predict import predict_video

st.set_page_config(page_title="Detection", page_icon="🔍", layout="wide")

with open("assets/cyber.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="df-card-plain">
  <h2 style="margin:0;">🔍 Detection</h2>
  <p style="color: rgba(230,230,230,.78); margin-top:6px;">
    Upload a short video clip for faster analysis. The system extracts audio, transcribes the script, then runs your trained classifier.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

colA, colB = st.columns([1.25, 1])

with colA:
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "m4v", "avi", "mkv"]
    )

    if uploaded:
        video_bytes = uploaded.getvalue()
        st.video(video_bytes)

with colB:
    st.markdown("""
<div class="df-card-plain">
<b>Pipeline</b>
<ul style="margin-top:8px;">
  <li>Audio extraction (FFmpeg)</li>
  <li>Script transcription (Whisper)</li>
  <li>Text + MFCC features (meta.json)</li>
  <li>Prediction (LogReg bundle)</li>
</ul>
<div class="df-hr"></div>
<b>Tip</b><br>
Use a 5–15 second clip for the fastest result.
</div>
""", unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.session_state["result"] = None

    run = st.button("🚀 Run Detection", use_container_width=True, disabled=(uploaded is None))

    if run and uploaded:
        with st.spinner("Extracting audio + transcribing + predicting..."):
            with tempfile.TemporaryDirectory() as td:
                vid_path = Path(td) / uploaded.name
                vid_path.write_bytes(video_bytes)
                result = predict_video(str(vid_path))

        st.session_state["result"] = result
        st.switch_page("pages/2_Results.py")
