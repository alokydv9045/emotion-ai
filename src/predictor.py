import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from src.config import MODEL_PATH, IMG_SIZE, EMOTIONS

model = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("[SUCCESS] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Model load error: {e}")
else:
    print(f"[WARNING] Model not found at: {MODEL_PATH}")
    print("   Run `python src/train.py` to train the model first.")

def predict_face(gray_face):
    """
    Predict emotion from a grayscale face crop.
    Returns (emotion_str, prob_2d_array) where prob_2d_array has shape (1, 7).
    """
    if model is None:
        return "Model Missing", np.zeros((1, len(EMOTIONS)))

    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    pred = model.predict(face, verbose=0)   # shape: (1, 7)
    emotion = EMOTIONS[np.argmax(pred[0])]

    return emotion, pred   # pred is (1,7) — consistent for all callers
