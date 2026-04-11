import cv2
import numpy as np
import os
import tensorflow as tf
import logging

# Disable TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CONFIG
IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
BACKEND_DIR = "backend"
MODEL_PATH = os.path.join(BACKEND_DIR, "models", "emotion_model_best.h5")
TEST_DATA_DIR = "dataset/test"

def build_precision_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input_layer")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", name="conv2d")(inputs)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", name="conv2d_1")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="max_pooling2d")(x)
    res_a = tf.keras.layers.SeparableConv2D(128, 3, padding="same", name="separable_conv2d")(x)
    res_a = tf.keras.layers.BatchNormalization(name="batch_normalization_2")(res_a)
    res_a = tf.keras.layers.ReLU(name="re_lu")(res_a)
    res_a = tf.keras.layers.SeparableConv2D(128, 3, padding="same", name="separable_conv2d_1")(res_a)
    res_a = tf.keras.layers.BatchNormalization(name="batch_normalization_3")(res_a)
    skip_a = tf.keras.layers.Conv2D(128, 1, padding="same", name="conv2d_2")(x)
    skip_a = tf.keras.layers.BatchNormalization(name="batch_normalization_4")(skip_a)
    x = tf.keras.layers.Add(name="add")([res_a, skip_a])
    x = tf.keras.layers.ReLU(name="re_lu_1")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="max_pooling2d_1")(x)
    res_b = tf.keras.layers.SeparableConv2D(256, 3, padding="same", name="separable_conv2d_2")(x)
    res_b = tf.keras.layers.BatchNormalization(name="batch_normalization_5")(res_b)
    res_b = tf.keras.layers.ReLU(name="re_lu_2")(res_b)
    res_b = tf.keras.layers.SeparableConv2D(256, 3, padding="same", name="separable_conv2d_3")(res_b)
    res_b = tf.keras.layers.BatchNormalization(name="batch_normalization_6")(res_b)
    skip_b = tf.keras.layers.Conv2D(256, 1, padding="same", name="conv2d_3")(x)
    skip_b = tf.keras.layers.BatchNormalization(name="batch_normalization_7")(skip_b)
    x = tf.keras.layers.Add(name="add_1")([res_b, skip_b])
    x = tf.keras.layers.ReLU(name="re_lu_3")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="max_pooling2d_2")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax", name="dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def run_benchmark(limit_per_emotion=20):
    print(f"--- Cortex-V Accuracy Benchmark (Limit: {limit_per_emotion}/emotion) ---")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    model = build_precision_model()
    model.load_weights(MODEL_PATH, by_name=True)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    results = {}
    total_found = 0
    total_correct = 0
    total_images = 0

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(TEST_DATA_DIR, emotion.lower())
        if not os.path.exists(emotion_dir):
            print(f"Warning: Category {emotion} missing in {TEST_DATA_DIR}")
            continue
        
        files = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg')]
        files = files[:limit_per_emotion]
        
        found = 0
        correct = 0
        
        for f in files:
            img_path = os.path.join(emotion_dir, f)
            frame = cv2.imread(img_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Since dataset images ARE faces (48x48), we usually skip detection or use very loose params
            # But the user wants to test the "system", so we test the whole pipeline
            faces = face_cascade.detectMultiScale(gray, 1.1, 1, minSize=(10, 10))
            
            # If detection fails on 48x48 (common for Haar), we force-process the whole image for accuracy check
            if len(faces) == 0:
                face_crop = gray
            else:
                found += 1
                total_found += 1
                x, y, w, h = faces[0]
                face_crop = gray[y:y+h, x:x+w]
            
            # Inferences
            norm_face = _clahe.apply(face_crop)
            resized = cv2.resize(norm_face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
            input_data = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            
            pred = model.predict(input_data, verbose=0)[0]
            pred_emotion = EMOTIONS[np.argmax(pred)]
            
            if pred_emotion.lower() == emotion.lower():
                correct += 1
                total_correct += 1
                
            total_images += 1
            
        results[emotion] = {
            "total": len(files),
            "detected": found,
            "accurate": correct,
            "acc_rate": (correct / len(files) * 100) if len(files) > 0 else 0
        }

    # Final Report
    print("\n" + "="*50)
    print(f"{'EMOTION':<12} | {'TOTAL':<6} | {'ACCURACY':<10}")
    print("-" * 50)
    for emotion, stats in results.items():
        print(f"{emotion:<12} | {stats['total']:<6} | {stats['acc_rate']:>8.1f}%")
    print("-" * 50)
    
    avg_acc = (total_correct / total_images * 100) if total_images > 0 else 0
    print(f"{'OVERALL':<12} | {total_images:<6} | {avg_acc:>8.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_benchmark(limit_per_emotion=30)
