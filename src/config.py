import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/emotion_model.h5")

# ── System Identity ──────────────────────────────────────────
SYSTEM_NAME = "Cortex-V"
TAGLINE     = "AI Jo Dil Ki Baat Jaan Le"

# ── Emotion Classes ──────────────────────────────────────────
EMOTIONS = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Neutral', 'Sad', 'Surprise'
]

# ── Hindi Emotion Labels ─────────────────────────────────────
EMOTIONS_HI = {
    'Angry':    'Gussa',
    'Disgust':  'Ghrina',
    'Fear':     'Darr',
    'Happy':    'Khushi',
    'Neutral':  'Shaant',
    'Sad':      'Dukhi',
    'Surprise': 'Hairani'
}

# ── ElevenLabs Acoustic Engine ───────────────────────────────
ELEVENLABS_API_KEY = ""        # Paste key from https://elevenlabs.io
USE_ELEVENLABS     = False     # Set True once key is added
DEFAULT_VOICE_ID   = "EXAVITQu4vr4xnSDxMaL"  # Bella (default)

# ── Offline Acoustic Engine (pyttsx3) ────────────────────────
VOICE_RATE   = 175
VOICE_VOLUME = 1.0
VOICE_GENDER = 'female'

# ── Hindi Voice Responses (gTTS-optimised, natural Hinglish) ─
VOICE_RESPONSES = {
    'Angry':    'Aap gusse mein hain. Gehri saans lijiye, sab theek ho jaayega.',
    'Disgust':  'Aap kuch pasand nahi kar rahe. Apna dhyan dusri cheez par lagaiye.',
    'Fear':     'Dariye mat. Cortex aapke saath hai, sab safe hai.',
    'Happy':    'Wah! Aap bahut khush lag rahe hain. Yeh muskaan banaye rakhiye!',
    'Neutral':  'Aap shaant hain. Yeh sukun ki avastha bahut achhi hai.',
    'Sad':      'Dil chhota mat keejiye. Har mushkil ke baad khushi aati hai.',
    'Surprise': 'Arre waah! Aap kaafi hairan lag rahe hain. Kya hua aaj?'
}

# ── Model Hyperparameters ────────────────────────────────────
IMG_SIZE      = 48
SMOOTH_WINDOW = 8
