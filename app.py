"""
Cortex-V  |  AI Emotion Engine — Pro UI
SaaS navbar · Glass panels · Scanner · Glowing Orb · Pro Metrics
"""
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image
import time
import plotly.graph_objects as go

try:
    from src.webcam   import get_faces
    from src.predictor import predict_face
    from src.smoothing import EmotionSmoother
    from src.voice     import speak, reset_last_emotion
    from src.config    import EMOTIONS, EMOTIONS_HI, SYSTEM_NAME, TAGLINE
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title=f"{SYSTEM_NAME} · AI Emotion Engine",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Emotion Tokens
# ──────────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "Happy":    "#22c55e",
    "Sad":      "#60a5fa",
    "Angry":    "#f87171",
    "Fear":     "#a78bfa",
    "Disgust":  "#fb923c",
    "Surprise": "#fbbf24",
    "Neutral":  "#94a3b8",
}
EMOTION_EMOJI = {
    "Happy": "😊", "Sad": "😢", "Angry": "😠",
    "Fear": "⚡", "Disgust": "🤢", "Surprise": "✨", "Neutral": "😐",
}

# ──────────────────────────────────────────────────────────────
# 🎨 GLOBAL DESIGN SYSTEM
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ---------- FONT ---------- */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif;
  background-color: #050714;
  color: #E5E7EB;
}

h1, h2, h3, .nav-logo {
  font-family: 'Outfit', sans-serif;
}

/* ---------- THEMES / VARS ---------- */
:root {
  --bg: #050714;
  --panel: rgba(13, 17, 33, 0.7);
  --border: rgba(255, 255, 255, 0.08);
  --accent: #22D3EE;
  --accent2: #8B5CF6;
  --glass: rgba(255, 255, 255, 0.03);
  --happy: #22C55E;
  --neutral: #94A3B8;
  --surprise: #FBBF24;
  --sad: #0EA5E9;
  --angry: #EF4444;
}

/* ---------- BACKGROUND ---------- */
.stApp {
  background: radial-gradient(circle at 50% -20%, rgba(34, 211, 238, 0.15), transparent 60%),
              radial-gradient(circle at 0% 100%, rgba(139, 92, 246, 0.1), transparent 50%),
              #050714;
}

/* ---------- HIDE DEFAULTS ---------- */
#MainMenu, footer, header, .stDeployButton, [data-testid="stToolbar"] {
  display: none !important;
}

/* ---------- NAVBAR ---------- */
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: rgba(13, 17, 33, 0.6);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 9999;
}

.nav-left { display: flex; align-items: center; gap: 12px; }
.nav-logo-box {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
}
.nav-logo { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; }
.nav-tagline { font-size: 10px; opacity: 0.5; text-transform: uppercase; letter-spacing: 1.5px; margin-top: -2px; }

.status-pill {
  background: rgba(34, 211, 238, 0.15);
  border: 1px solid rgba(34, 211, 238, 0.3);
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  display: flex; align-items: center; gap: 6px;
  color: var(--accent);
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 10px var(--accent); }

/* ---------- GLASS CARD ---------- */
.glass-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  backdrop-filter: blur(12px);
  margin-bottom: 20px;
}

.card-title {
  font-size: 11px;
  font-weight: 700;
  color: rgba(255,255,255,0.4);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 15px;
  display: flex; align-items: center; gap: 8px;
}

/* ---------- SCANNER HUD ---------- */
.scanner-container {
  position: relative;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(34, 211, 238, 0.2);
  background: #000;
}

.scanner-hud {
  position: absolute;
  bottom: 12px; left: 12px;
  font-family: monospace;
  font-size: 10px;
  color: rgba(255,255,255,0.6);
  display: flex; gap: 15px;
  z-index: 10;
}

.rec-label {
  position: absolute;
  top: 15px; right: 20px;
  background: rgba(0,0,0,0.5);
  padding: 4px 10px;
  border-radius: 4px;
  display: flex; align-items: center; gap: 6px;
  font-size: 10px; font-weight: 700;
  z-index: 10;
}
.rec-dot { width: 8px; height: 8px; background: #EF4444; border-radius: 50%; animation: blink 1s infinite; }

@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* ---------- ANALYTICS SIDEBAR ---------- */
.sidebar-title {
  font-size: 12px;
  font-weight: 700;
  color: rgba(255,255,255,0.5);
  margin-bottom: 20px;
  letter-spacing: 1px;
}

.stat-item {
  margin-bottom: 20px;
}
.stat-header {
  display: flex; justify-content: space-between;
  font-size: 12px; margin-bottom: 6px;
}
.stat-bar-bg {
  height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden;
}
.stat-bar-fill {
  height: 100%; border-radius: 3px;
  transition: width 0.5s ease;
}

/* ---------- STABILITY GAUGE ---------- */
.gauge-container {
  position: relative;
  width: 100px; height: 100px;
  margin-bottom: 15px;
}
.gauge-svg { transform: rotate(-90deg); }
.gauge-bg { fill: none; stroke: rgba(255,255,255,0.05); stroke-width: 8; }
.gauge-fill {
  fill: none; stroke: var(--accent); stroke-width: 8;
  stroke-linecap: round; stroke-dasharray: 251.2;
  transition: stroke-dashoffset 0.5s ease;
}
.gauge-text {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; font-weight: 700; color: white;
}

/* ---------- INSIGHT BOX ---------- */
.insight-box {
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.1), rgba(139, 92, 246, 0.1));
  border: 1px solid rgba(34, 211, 238, 0.3);
  border-radius: 12px;
  padding: 15px;
  display: flex; gap: 12px;
  margin-top: 15px;
}
.insight-icon {
  width: 36px; height: 36px; border-radius: 50%;
  background: rgba(34, 211, 238, 0.2);
  display: flex; align-items: center; justify-content: center;
  color: var(--accent);
}
.insight-content h4 { font-size: 11px; margin: 0; color: var(--accent); text-transform: uppercase; letter-spacing: 1px; }
.insight-content p { font-size: 12px; margin: 4px 0 0; opacity: 0.8; line-height: 1.4; }

/* ---------- SIDEBAR (CONTROL PANEL) ---------- */
[data-testid="stSidebar"] {
  background-color: #03050F !important;
  border-right: 1px solid var(--border) !important;
  padding-top: 0 !important;
}

.sb-header {
  font-size: 11px; font-weight: 700; color: rgba(255,255,255,0.3);
  padding: 24px 0 16px; letter-spacing: 1.5px;
}

/* ---------- TAB STYLING ---------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 20px;
  background: transparent;
  padding: 0;
  border-bottom: 2px solid rgba(255,255,255,0.05);
  margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
  height: 40px;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  color: rgba(255,255,255,0.4) !important;
  font-size: 14px !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
  width: 100%; border-radius: 8px; border: 1px solid var(--border);
  background: rgba(255,255,255,0.03); color: white; padding: 10px;
  transition: all 0.2s;
}
.stButton>button:hover {
  border-color: var(--accent); background: rgba(34, 211, 238, 0.05);
}

/* ---------- SLIDER / TOGGLE ---------- */
.stSlider [data-testid="stTickBar"] { display: none; }
.stSlider label { font-size: 13px !important; color: #94A3B8 !important; }

/* ---------- OVERRIDES ---------- */
.stApp { padding-top: 70px !important; }
[data-testid="stColumn"] { padding: 0 !important; }

</style>

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Lucide Icons CDN
# ──────────────────────────────────────────────────────────────
components.html("""
<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>
<script>
  function ic(){if(window.parent&&window.parent.document){
    var s=window.parent.document.querySelector('script[data-lc]');
    if(!s){s=window.parent.document.createElement('script');
      s.src='https://unpkg.com/lucide@latest/dist/umd/lucide.min.js';
      s.dataset.lc='1';
      s.onload=function(){[0,500,1500].forEach(function(d){setTimeout(function(){if(window.parent.lucide)window.parent.lucide.createIcons();},d);});};
      window.parent.document.head.appendChild(s);}
  }}
  ic();
  [400,1200,3000].forEach(function(d){setTimeout(ic,d);});
</script>
""", height=0)

# ──────────────────────────────────────────────────────────────
# 🧭 TOP NAVBAR
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="navbar">
  <div class="nav-left">
    <div class="nav-logo-box">
      <i data-lucide="brain-circuit" style="stroke:#fff; width:20px; height:20px;"></i>
    </div>
    <div>
      <div class="nav-logo">{SYSTEM_NAME}</div>
      <div class="nav-tagline">AI JO DIL KI BAAT JAAN LE</div>
    </div>
  </div>
  
  <div class="status-pill">
    <div class="status-dot"></div>
    Camera Ready
  </div>

  <div style="display:flex; align-items:center; gap:15px; opacity:0.8;">
    <i data-lucide="moon" style="width:18px; height:18px; cursor:pointer;"></i>
    <div style="width:32px; height:32px; background:var(--accent); border-radius:8px; display:flex; align-items:center; justify-content:center; cursor:pointer;">
      <i data-lucide="mic" style="stroke:#000; width:18px; height:18px;"></i>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Sidebar (Control Panel)
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-header">AI CONTROL PANEL</div>', unsafe_allow_html=True)

    voice_on = st.toggle("Voice Feedback", value=True, key="voice")
    
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="sb-lbl">Emotion Sensitivity</div>', unsafe_allow_html=True)
    sensitivity = st.slider("High", 1, 15, 8, key="smooth", label_visibility="visible")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="sb-lbl">Language Model</div>', unsafe_allow_html=True)
    lang = st.selectbox("", ["English / Hindi"], label_visibility="collapsed")

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="sb-lbl">Webcam Resolution</div>', unsafe_allow_html=True)
    res = st.selectbox("", ["1080p HD", "720p SD", "4k Ultra"], label_visibility="collapsed")

    st.markdown('<div style="height:100px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sep" style="margin-bottom:20px;"></div>', unsafe_allow_html=True)
    adv_analytics = st.toggle("Advanced Analytics", value=True)

# ──────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────────────────────
if "n_detections"     not in st.session_state: st.session_state.n_detections     = 0
if "n_frames"         not in st.session_state: st.session_state.n_frames         = 0
if "session_start"    not in st.session_state: st.session_state.session_start    = time.time()
if "emotion_history"  not in st.session_state: st.session_state.emotion_history  = []
if "stability_score"  not in st.session_state: st.session_state.stability_score  = 92

smoother = EmotionSmoother(size=sensitivity)

# ──────────────────────────────────────────────────────────────
# 🎨 DASHBOARD RENDERERS
# ──────────────────────────────────────────────────────────────

def render_stability(score, status="High Stability"):
    offset = 251.2 - (251.2 * score / 100)
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:20px;">
        <div class="gauge-container">
            <svg class="gauge-svg" viewBox="0 0 100 100">
                <circle class="gauge-bg" cx="50" cy="50" r="40" />
                <circle class="gauge-fill" cx="50" cy="50" r="40" style="stroke-dashoffset: {offset};" />
            </svg>
            <div class="gauge-text">{int(score)}</div>
        </div>
        <div>
            <div style="font-size:14px; font-weight:700; color:white;">{status}</div>
            <div style="font-size:11px; opacity:0.5;">Consistent emotional state detected.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_emotion_bars(probs):
    html = ""
    for i, (emo, p) in enumerate(zip(EMOTIONS, probs)):
        color = EMOTION_COLORS.get(emo, "#22D3EE")
        val = p * 100
        html += f"""
        <div class="stat-item">
            <div class="stat-header">
                <span>{emo}</span>
                <span style="color:{color}; font-weight:700;">{val:.1f}%</span>
            </div>
            <div class="stat-bar-bg">
                <div class="stat-bar-fill" style="width:{val}%; background:{color};"></div>
            </div>
        </div>
        """
    bars_slot.markdown(html, unsafe_allow_html=True)

def render_insight(emotion):
    msgs = {
        "Happy": "You seem genuinely happy right now. Keep that positive energy! 😊",
        "Neutral": "You are maintaining a calm and composed state. 🧘",
        "Surprise": "Something caught your attention! Your expression shows high engagement. ✨",
        "Sad": "We detected a dip in your mood. Take a deep breath, you're doing great. 💙",
        "Angry": "Intense emotions detected. Try to take a moment to relax. 🧘‍♂️",
    }
    msg = msgs.get(emotion, "System is analyzing your emotional patterns.")
    insight_slot.markdown(f"""
    <div class="insight-box">
        <div class="insight-icon"><i data-lucide="sparkles"></i></div>
        <div class="insight-content">
            <h4>CORTEX-V INSIGHT</h4>
            <p>{msg}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_history_chart():
    history = st.session_state.emotion_history[-50:] if st.session_state.emotion_history else []
    if not history: return
    
    fig = go.Figure(go.Scatter(
        y=history, mode='lines', 
        line=dict(color='#22D3EE', width=3),
        fill='tozeroy', 
        fillcolor='rgba(34, 211, 238, 0.1)'
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=180, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False
    )
    history_slot.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def render_freq_gauge():
    val = 85 # Placeholder or logic based on detections
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        number = {'suffix': "%", 'font': {'color': 'white', 'size': 30}},
        gauge = {
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': "#22C55E"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", height=180,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    freq_slot.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ──────────────────────────────────────────────────────────────
# 🧩 MAIN LAYOUT
# ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Live Emotion Scan", "Image Emotion Analysis"])

with tab1:
    left, right = st.columns([2.2, 1], gap="medium")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Scanner HUD Container
        st.markdown("""
        <div class="scanner-container">
            <div class="rec-label">
                <div class="rec-dot"></div> REC
            </div>
            <div class="scanner-hud">
                <span>FPS: 59.94</span>
                <span>RES: 1080p</span>
                <span>LATENCY: 12ms</span>
            </div>
        """, unsafe_allow_html=True)
        
        frame_slot = st.empty()
        st.markdown('</div>', unsafe_allow_html=True) # close scanner-container
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        # Bottom area in Left col
        b_left, b_right = st.columns(2)
        with b_left:
            st.markdown('<div class="glass-card" style="min-height:220px;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i data-lucide="activity"></i> Emotion History (Last 5 mins)</div>', unsafe_allow_html=True)
            history_slot = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with b_right:
            st.markdown('<div class="glass-card" style="min-height:220px;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i data-lucide="clock"></i> Session Frequency</div>', unsafe_allow_html=True)
            freq_slot = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True) # close outer glass-card

    with right:
        st.markdown('<div class="sidebar-title">REAL-TIME ANALYTICS</div>', unsafe_allow_html=True)
        
        # Stability Score
        st.markdown('<div class="sb-header">STABILITY SCORE</div>', unsafe_allow_html=True)
        stability_slot = st.empty()
        
        # Emotion Bars
        st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
        bars_slot = st.empty()
        
        # Insight Notification
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        insight_slot = st.empty()

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">IMAGE EMOTION ENGINE</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","png","jpeg"], label_visibility="collapsed", key="uploader")
    if uploaded:
        pil_img    = Image.open(uploaded)
        upload_arr = np.array(pil_img)
        upload_bgr = cv2.cvtColor(upload_arr, cv2.COLOR_RGB2BGR)
        st.image(pil_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Live Webcam Loop
# ──────────────────────────────────────────────────────────────
with left:
    run = st.toggle("Initialize Engine", value=False, key="run_engine")

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Camera unavailable — check permissions and refresh.")
    else:
        reset_last_emotion()
        fps_timer = time.time()
        try:
            while run:
                ret, frame = cap.read()
                if not ret: break

                st.session_state.n_frames += 1
                curr_time = time.time()
                fps = 1 / (curr_time - fps_timer) if (curr_time - fps_timer) > 0 else 30
                fps_timer = curr_time

                gray, faces = get_faces(frame)
                
                # Default empty state
                if len(faces) == 0:
                    render_emotion_bars([0]*7)
                    render_stability(st.session_state.stability_score)
                
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    raw_em, prob = predict_face(face)
                    emotion  = smoother.update(raw_em)
                    color_hex = EMOTION_COLORS.get(emotion, "#22D3EE")
                    color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0)) # Hex to BGR
                    
                    st.session_state.n_detections += 1
                    
                    # Update History
                    emo_idx = list(EMOTIONS).index(emotion)
                    st.session_state.emotion_history.append(emo_idx)
                    
                    # Dashboard Updates
                    render_emotion_bars(prob[0])
                    render_stability(st.session_state.stability_score)
                    render_insight(emotion)
                    render_history_chart()
                    render_freq_gauge()

                    if voice_on: speak(emotion)

                    # HUD Face Box (Match Design)
                    conf = prob[0][emo_idx] * 100
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
                    
                    # Label background
                    label = f"{emotion.upper()} {conf:.1f}%"
                    (l_w, l_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x, y-25), (x + l_w + 10, y), color_bgr, -1)
                    cv2.putText(frame, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_slot.image(frame_rgb, use_container_width=True)
                
                # Stability logic (mock for now, or based on variance)
                if len(st.session_state.emotion_history) > 10:
                    recent = st.session_state.emotion_history[-10:]
                    st.session_state.stability_score = 100 - (np.std(recent) * 20)
                    st.session_state.stability_score = max(0, min(100, st.session_state.stability_score))

        finally:
            cap.release()

# ──────────────────────────────────────────────────────────────
# Image Upload Detection
# ──────────────────────────────────────────────────────────────
elif uploaded:
    gray, faces = get_faces(upload_bgr)
    if faces is not None and len(faces) > 0:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            emotion, prob = predict_face(face)
            
            # Dashboard Updates
            render_emotion_bars(prob[0])
            render_insight(emotion)
            
            if voice_on: speak(emotion)
    else:
        st.warning("No faces detected in this image.")
