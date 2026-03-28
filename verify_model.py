import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/emotion_model.h5")

def verify():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy image (48x48 grayscale)
    dummy_img = np.zeros((1, 48, 48, 1), dtype=np.float32)
    
    print("Running prediction on dummy image...")
    try:
        prediction = model.predict(dummy_img, verbose=0)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction: {prediction}")
        print("Verification Successful!")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    verify()
