import json
import streamlit as st

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")

with open("assets/cyber.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Results")

r = st.session_state.get("result")
if not r:
    st.warning("No result found. Go to Detection and run analysis.")
    st.stop()

pred = str(r.get("prediction", "UNKNOWN")).upper()
conf = float(r.get("confidence", 0.0)) * 100.0
p_real = float(r.get("prob_real", 0.0)) * 100.0
p_fake = float(r.get("prob_fake", 0.0)) * 100.0
transcript = r.get("transcript", "") or ""

badge_class = "fake" if pred == "FAKE" else "real"

st.markdown(f"""
<div class="df-card">
  <div class="df-badge {badge_class}">VERDICT: {pred}</div>
  <div class="df-kpi">
    <div class="item"><div class="label">Confidence (predicted class)</div><div class="value">{conf:.2f}%</div></div>
    <div class="item"><div class="label">REAL probability</div><div class="value">{p_real:.2f}%</div></div>
    <div class="item"><div class="label">FAKE probability</div><div class="value">{p_fake:.2f}%</div></div>
  </div>
  <div class="df-hr"></div>
  <b>Explanation</b><br>
  <span style="color: rgba(230,230,230,.78);">
    Prediction is computed using your saved training configuration (meta.json) with TF-IDF (script) + MFCC (audio) features.
  </span>
</div>
""", unsafe_allow_html=True)

st.markdown("")
st.subheader("📝 Script / Transcription")
st.markdown(f"<div class='df-card-plain df-mono'>{transcript if transcript else 'No transcript extracted.'}</div>", unsafe_allow_html=True)

st.markdown("")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔁 Analyze another file", use_container_width=True):
        st.session_state["result"] = None
        st.switch_page("pages/1_Detection.py")

with col2:
    report = {
        "prediction": r.get("prediction"),
        "confidence": r.get("confidence"),
        "prob_real": r.get("prob_real"),
        "prob_fake": r.get("prob_fake"),
        "transcript": r.get("transcript"),
    }
    st.download_button(
        "⬇️ Download Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="deepfake_report.json",
        mime="application/json",
        use_container_width=True
    )
