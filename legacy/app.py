"""
Cortex-V (Lean)  |  Focused Emotion AI
Simple · Fast · Stable
"""
import streamlit as st
import cv2
import numpy as np
import base64
import time
from PIL import Image
import os

from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import add_script_run_context, get_script_run_ctx

# ── IMPORTS & SAFETY ──
try:
    from src.webcam    import get_faces, ThreadedWebcam
    from src.predictor import predict_face, HARDWARE_STATUS
    from src.smoothing import EmotionSmoother
    from src.config    import EMOTIONS, SYSTEM_NAME, TAGLINE
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ── GLOBAL EXECUTOR (AI Thread Pool) ──
if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=1)

# ── HARDWARE LIFECYCLE ──
@st.cache_resource(show_spinner="Connecting to Hardware...")
def get_hardware_link():
    """Returns a started ThreadedWebcam instance."""
    try:
        hw = ThreadedWebcam(0)
        return hw.start()
    except Exception as e:
        print(f"[Hardware] Init error: {e}")
        return None

def release_hardware():
    try:
        hw = get_hardware_link()
        if hw: hw.stop()
        st.cache_resource.clear()
    except: pass

# ── UTILITIES ──
def get_image_base64(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return base64.b64encode(buffer).decode()
    except: return None

# ── PAGE SETUP ──
st.set_page_config(layout="centered", page_title=f"{SYSTEM_NAME}", page_icon="⚡")

# ── CSS (Premium Clean UI) ──
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700&family=Inter:wght@400;500&display=swap');

:root {{
    --accent: #22D3EE;
    --bg-deep: #05070A;
    --panel: rgba(15, 23, 42, 0.6);
    --border: rgba(255, 255, 255, 0.08);
}}

html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-deep);
    color: #F8FAFC;
}}

.main-title {{
    font-family: 'Outfit', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #FFF 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0px;
}}

.tagline {{
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: #64748B;
    text-align: center;
    margin-bottom: 2.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    justify-content: center;
    background-color: transparent;
}}

.stTabs [data-baseweb="tab"] {{
    height: 45px;
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px 8px 0px 0px;
    padding: 0px 24px;
    color: #94A3B8;
}}

.stTabs [aria-selected="true"] {{
    background-color: rgba(34, 211, 238, 0.1) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}}

.result-card {{
    background: var(--panel);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}}

.emotion-label {{
    font-family: 'Outfit', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--accent);
    margin: 10px 0;
    text-shadow: 0 0 20px rgba(34, 211, 238, 0.4);
}}

.confidence-bar {{
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    margin-top: 15px;
    overflow: hidden;
}}

.confidence-fill {{
    height: 100%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent);
}}

.stream-container {{
    border-radius: 16px;
    border: 1px solid var(--border);
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0,0,0,0.6);
}}

/* Clean up Streamlit elements */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stHeader"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="main-title">{SYSTEM_NAME}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="tagline">{TAGLINE}</div>', unsafe_allow_html=True)

# ── SESSION STATE ──
if "smoother" not in st.session_state:
    st.session_state.smoother = EmotionSmoother(size=5)
if "last_ai_result" not in st.session_state:
    st.session_state.last_ai_result = ("Neutral", 0, []) # emotion, confidence, faces
if "ai_busy" not in st.session_state:
    st.session_state.ai_busy = False

# ── MAIN TABS ──
tab1, tab2 = st.tabs(["LIVE SCANNER", "PHOTO ANALYSIS"])

def run_ai_task(frame):
    """Background AI logic: detect faces and predict emotion."""
    gray, faces = get_faces(frame)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        raw_emo, prob = predict_face(face)
        # Smoother update remains in main thread to avoid context issues
        return raw_emo, prob, faces
    return "Neutral", np.zeros((1, 1)), []

with tab1:
    col1, col2 = st.columns([1.8, 1], gap="medium")
    
    with col1:
        stream_slot = st.empty()
    with col2:
        st.markdown('<div style="padding-top: 10px;"></div>', unsafe_allow_html=True)
        run = st.toggle("ACTIVATE SENSOR", value=False)
        status_slot = st.empty()

    @st.fragment(run_every=0.04) # Ultra-fast visual update (25 FPS)
    def live_engine():
        if not run:
            release_hardware()
            status_slot.markdown('<div class="result-card" style="color:#64748B;">SENSOR OFFLINE</div>', unsafe_allow_html=True)
            return

        hw = get_hardware_link()
        if not hw or not hw.started:
            st.error("HARDWARE LINK FAILED")
            return

        ret, frame = hw.read()
        if ret and frame is not None:
            # 1. TRIGGER ASYNC AI (If not busy)
            if not st.session_state.ai_busy:
                st.session_state.ai_busy = True
                
                # Capture current context
                ctx = get_script_run_ctx()

                def wrapped_task(f):
                    add_script_run_context(threading.current_thread(), ctx)
                    return run_ai_task(f)

                def callback(future):
                    raw_emo, prob, fs = future.result()
                    # Process result safely
                    if fs:
                        active_emo = st.session_state.smoother.update(raw_emo)
                        conf = np.max(prob) * 100
                        st.session_state.last_ai_result = (active_emo, conf, fs)
                    st.session_state.ai_busy = False

                import threading
                future = st.session_state.executor.submit(wrapped_task, frame.copy())
                future.add_done_callback(callback)

            # 2. RENDER FEED (Instant)
            active_emo, confidence, faces = st.session_state.last_ai_result
            
            # Draw HUD from best-known face location
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (238, 211, 34), 2)
            
            b64 = get_image_base64(frame)
            if b64:
                stream_slot.markdown(f'<div class="stream-container"><img src="data:image/jpeg;base64,{b64}" style="width:100%;"></div>', unsafe_allow_html=True)
            
            # 3. RENDER STATUS (Instant)
            status_slot.markdown(f"""
                <div class="result-card">
                    <div style="font-size:11px; color:#64748B; letter-spacing:1px;">LATEST DETECTION</div>
                    <div class="emotion-label">{active_emo.upper()}</div>
                    <div style="font-size:12px; color:var(--accent);">{confidence:.1f}% Match</div>
                    <div class="confidence-bar"><div class="confidence-fill" style="width:{confidence}%;"></div></div>
                </div>
            """, unsafe_allow_html=True)

    live_engine()

with tab2:
    st.markdown('<div style="padding: 20px 0;"></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CHOOSE IMAGE FOR ANALYSIS", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("SCANNING NEURAL LAYERS..."):
            gray, faces = get_faces(img)
            
            if not faces:
                st.warning("NO FACIAL DATA DETECTED")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
            else:
                for (x, y, w, h) in faces:
                    face_crop = gray[y:y+h, x:x+w]
                    label, prob = predict_face(face_crop)
                    conf = np.max(prob) * 100
                    
                    # Pro Draw
                    cv2.rectangle(img, (x, y), (x+w, y+h), (34, 211, 238), 2)
                    
                    st.markdown(f"""
                        <div class="result-card" style="margin-bottom: 20px;">
                            <div style="font-size:11px; color:#64748B; letter-spacing:1px;">SPECTRAL RESULT</div>
                            <div class="emotion-label">{label.upper()}</div>
                            <div style="font-size:12px; color:var(--accent);">{conf:.1f}% Match</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<div class="stream-container">', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──
st.markdown('<div style="margin-top: 50px; border-top: 1px solid var(--border); padding-top: 20px; text-align:center; font-size:10px; color:#475569; letter-spacing:1px;">CORTEX-V LEAN ENGINE V2.0 // HIGH-SPEED EMOTION INTELLIGENCE</div>', unsafe_allow_html=True)
