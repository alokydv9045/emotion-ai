import os

# ── CORTEX-V IDENTITY ──
SYSTEM_NAME = "CORTEX-V"
TAGLINE     = "NEURAL EMOTION INTELLIGENCE // LEAN ENGINE"

# ── ENGINE PATHS ──
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "..", "models", "emotion_model.h5")

# ── LOGIC TOKENS ──
EMOTIONS    = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE    = 48

# ── PERFORMANCE ──
UI_FPS      = 25
AI_THROTTLE = 0.5 
