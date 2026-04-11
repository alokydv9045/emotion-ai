import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import threading
from collections import deque
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

# ── HARDWARE LOCKDOWN ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("VRAM Dynamic Growth Enabled.")
    except RuntimeError as e:
        logging.error(f"VRAM Lock Error: {e}")

# ── LOGGING ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CortexBackend")

tf.keras.backend.set_image_data_format('channels_last')

# ── CONFIG ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use the peak-accuracy 'best' model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "emotion_model_best.h5")
IMG_SIZE = 96 # Phase 2 Resolution
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

app = FastAPI(title="Cortex-V AI Backend")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ELEVENLABS ENGINE ──
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY) if ELEVEN_API_KEY else None

# Voice ID: Bella (Expressive & Premium) - Can be changed to a custom Hindi voice if needed
VOICE_ID = "pNInz6obpgDQGcFmaJgB" # Custom Voice (Adam)

EMOTION_VOICE_SETTINGS = {
    "Angry": VoiceSettings(stability=0.2, similarity_boost=0.8, style=0.5, use_speaker_boost=True),
    "Happy": VoiceSettings(stability=0.4, similarity_boost=0.7, style=0.3, use_speaker_boost=True),
    "Sad": VoiceSettings(stability=0.3, similarity_boost=0.7, style=0.2, use_speaker_boost=True),
    "Surprise": VoiceSettings(stability=0.2, similarity_boost=0.9, style=0.4, use_speaker_boost=True),
    "Neutral": VoiceSettings(stability=0.7, similarity_boost=0.5, style=0.0, use_speaker_boost=True),
    "Fear": VoiceSettings(stability=0.2, similarity_boost=0.8, style=0.6, use_speaker_boost=True),
    "Disgust": VoiceSettings(stability=0.3, similarity_boost=0.7, style=0.4, use_speaker_boost=True),
}

# ── HYPER-ARCHITECTURE (V2) ──
# Import directly from train_engine to ensure 100% architecture parity for weights
import sys
sys.path.append(BASE_DIR)
# ── APEX-V5 ARCHITECTURE ──
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation('relu')(x)

def build_apex_v5_model():
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input_layer")
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)
    x = tf.keras.layers.GlobalAveragePooling2D(name="apex_gap")(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5, name="apex_dropout")(x)
    outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax", name="apex_emotion_softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


# Load weights
model = None
@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = build_apex_v5_model()
        model(np.zeros((1, IMG_SIZE, IMG_SIZE, 1))) 
        if os.path.exists(MODEL_PATH):
            model.load_weights(MODEL_PATH, by_name=True)

            logger.info("Apex v5.0 (RESIDUAL) Engine Online.")
        else:
            logger.warning("Apex Phase 2 Weights missing. Training mode recommended.")
    except Exception as e:
        logger.error(f"Inference Initialization Error: {e}")
        model = None



# ── API MODELS ──
class EmotionRequest(BaseModel):
    image: str # Base64 string
    stream: bool = True # Toggle temporal smoothing
    session_id: str = "default"

class SpeakRequest(BaseModel):
    text: str
    emotion: str = "Neutral"

# Temporal Smoothing Buffer (5 frames)
temporal_buffer = deque(maxlen=5)
inference_lock = threading.Lock()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ── ROBUST CASCADE LOAD ──
face_cascades = [
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
]

# ── ROUTES ──
@app.get("/health")
def health():
    return {"status": "online", "engine": "ready" if model else "failed", "v": "5.0_APEX"}

@app.post("/predict")
async def predict(req: EmotionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Neural engine offline")
    
    try:
        # 1. Decode Base64
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        data = base64.b64decode(encoded)
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        h_img, w_img = frame.shape[:2]

        # 2. Robust Multi-Cascade Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        for cascade in face_cascades:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            if len(faces) > 0:
                break
        
        # Emergency Fallback: If no face found but image is small portrait-like, treat whole image as face
        if len(faces) == 0:
            if not req.stream:
                # For tactical photo scans, use the center-crop as a fallback face if detection fails
                h, w = gray.shape
                # Only if the image is reasonably square/portrait
                if 0.5 < (w/h) < 2.0:
                    faces = [[int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8)]]
                    logger.info("TACTICAL FALLBACK: Using center-crop as face.")
        
        if len(faces) == 0:
            return {
                "emotion": "Neutral", 
                "confidence": 0, 
                "scores": {emo: 0.0 for emo in EMOTIONS},
                "faces": [], 
                "img_width": w_img, 
                "img_height": h_img
            }

        # 3. Process Primary Face (Protected by Thread Lock)
        with inference_lock:
            (x, y, w, h) = faces[0]
            face_crop = gray[max(0, y):min(h_img, y+h), max(0, x):min(w_img, x+w)]
            
            # 4. Neural Pre-processing (CLAHE + Raw Pixel Alignment)
            norm_face = clahe.apply(face_crop)
            resized = cv2.resize(norm_face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
            input_data = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            # 5. Inference
            pred_tensor = model(input_data, training=False)
            raw_scores = pred_tensor.numpy()[0]
            
            # 6. Session-Aware Smoothing & Isolation
            if req.stream:
                temporal_buffer.append(raw_scores)
                smoothed_scores = np.mean(list(temporal_buffer), axis=0)
            else:
                # Instant result for one-off uploads, purge buffer to isolate session
                temporal_buffer.clear()
                smoothed_scores = raw_scores
            
            emotion_idx = np.argmax(smoothed_scores)
            confidence = float(smoothed_scores[emotion_idx] * 100)
            
            res_scores = {CATEGORIES[i]: float(smoothed_scores[i] * 100) for i in range(len(CATEGORIES))}

        return {
            "emotion": CATEGORIES[emotion_idx],
            "confidence": confidence,
            "scores": res_scores,
            "faces": faces.tolist() if isinstance(faces, np.ndarray) else faces,
            "img_width": w_img,
            "img_height": h_img
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/speak")
async def speak(req: SpeakRequest):
    logger.info(f"VOICE_ENGINE: Process request for emotion={req.emotion}")
    if not eleven_client:
        logger.error("VOICE_ENGINE: Client offline")
        raise HTTPException(status_code=503, detail="ElevenLabs integration offline (Check API Key)")
    
    try:
        # 1. Map Emotion to Settings
        settings = EMOTION_VOICE_SETTINGS.get(req.emotion, EMOTION_VOICE_SETTINGS["Neutral"])
        logger.info(f"VOICE_ENGINE: Using settings for {req.emotion}")
        
        # 2. Add Emotional Tags for Human-like nuance (multilingual v2/flash support)
        # Note: Flash v2.5 handles context very well.
        emotion_tags = {
            "Angry": "[angry]",
            "Happy": "[cheerful]",
            "Sad": "[sad]",
            "Surprise": "[surprised]",
            "Fear": "[terrified]",
            "Neutral": ""
        }
        tag = emotion_tags.get(req.emotion, "")
        full_text = f"{tag} {req.text}"
        
        # 3. Generate Audio (Instant Flash v2.5)
        # We use eleven_flash_v2_5 for sub-second latency
        # Correct SDK method for v2.0+: client.text_to_speech.convert
        audio_gen = eleven_client.text_to_speech.convert(
            text=full_text,
            voice_id=VOICE_ID,
            model_id="eleven_flash_v2_5", 
            voice_settings=settings
        )
        
        # 4. Convert generator to bytes for base64 return
        audio_bytes = b"".join(list(audio_gen))
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return JSONResponse(content={"audio": audio_base64})
        
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice Engine Error: {str(e)}")

# ── PRODUCTION STATIC SERVING ──
DIST_PATH = os.path.join(BASE_DIR, "frontend", "dist")

if os.path.exists(DIST_PATH):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_PATH, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        if full_path.startswith("predict") or full_path.startswith("health"):
            pass 
        
        index_file = os.path.join(DIST_PATH, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        
        raise HTTPException(status_code=404, detail="Static assets missing")
else:
    logger.warning("Production DIST folder missing. Web UI will not be served from backend.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, limit_concurrency=10, timeout_keep_alive=5)
