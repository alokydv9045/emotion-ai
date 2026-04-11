import sys
import os
import cv2
import numpy as np
import tensorflow as tf

# Fix pathing to allow legacy import
sys.path.append(os.path.join(os.getcwd(), "legacy"))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.predictor import build_precision_model

# CONFIG
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_PATH = "backend/models/emotion_model.h5"
TEST_DATA_DIR = "dataset/test"

def run_tan_h_benchmark(limit_per_emotion=10):
    print(f"--- TanH [-1, 1] Normalization Accuracy Check ---")
    
    model = build_precision_model()
    model.load_weights(MODEL_PATH, by_name=True)

    total = 0
    correct = 0

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(TEST_DATA_DIR, emotion.lower())
        files = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg')][:limit_per_emotion]
        
        for f in files:
            img_path = os.path.join(emotion_dir, f)
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # [-1, 1] Normalization
            resized = cv2.resize(frame, (48, 48)).astype("float32")
            normalized = (resized - 127.5) / 127.5
            input_data = normalized.reshape(1, 48, 48, 1)
            
            pred = model.predict(input_data, verbose=0)[0]
            pred_emotion = EMOTIONS[np.argmax(pred)]
            
            if pred_emotion.lower() == emotion.lower():
                correct += 1
            total += 1
            
    print("\n" + "="*30)
    print(f"TanH [-1, 1] TOTAL ACC: {(correct/total)*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    run_tan_h_benchmark()
