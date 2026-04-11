import sys
import os
import cv2
import numpy as np
import tensorflow as tf

# Fix pathing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BACKEND_DIR = "backend"
MODEL_PATH = os.path.join(BACKEND_DIR, "models", "emotion_model_best.h5")
TEST_DATA_DIR = "dataset/test"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE = 48

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

def run_no_clahe_benchmark(limit_per_emotion=20):
    print(f"--- NO-CLAHE Final Audit (Limit: {limit_per_emotion}/emotion) ---")
    model = build_precision_model()
    model.load_weights(MODEL_PATH, by_name=True)

    results = {}
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(TEST_DATA_DIR, emotion.lower())
        files = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg')][:limit_per_emotion]
        correct = 0
        for f in files:
            img_path = os.path.join(emotion_dir, f)
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
            input_data = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            pred = model.predict(input_data, verbose=0)[0]
            if EMOTIONS[np.argmax(pred)].lower() == emotion.lower():
                correct += 1
        results[emotion] = (correct, len(files))

    print("\n" + "="*50)
    print(f"{'EMOTION':<12} | {'TOTAL':<6} | {'ACCURACY':<10}")
    print("-" * 50)
    total_samples = 0
    total_correct = 0
    for emotion, (correct, total) in results.items():
        acc = (correct/total)*100 if total > 0 else 0
        print(f"{emotion:<12} | {total:<6} | {acc:>9.1f}%")
        total_samples += total
        total_correct += correct
    print("-" * 50)
    print(f"{'OVERALL':<12} | {total_samples:<6} | {(total_correct/total_samples)*100:>9.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_no_clahe_benchmark()
