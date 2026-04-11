import cv2
import numpy as np
import os
import tensorflow as tf

# Load the model logic directly
IMG_SIZE = 48
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_PATH = "backend/models/emotion_model.h5"

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
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="max_pooling2d_2")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax", name="dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

print("Loading model...")
model = build_precision_model()
model.load_weights(MODEL_PATH, by_name=True)

# Test with a known "happy" image from dataset
test_img_path = "dataset/test/happy/PrivateTest_10077120.jpg"
frame = cv2.imread(test_img_path)
if frame is None:
    print(f"Failed to load {test_img_path}")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# TEST 1: New Robust Parameters (No Downsample, 1.1, 3, 20x20)
h_img, w_img = gray.shape[:2]
faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
print(f"Test 1 (New Robust Params): Detected {len(faces)} faces in {w_img}x{h_img} image.")

# TEST 2: Robust parameter set (No Downsample, scaleFactor 1.3)
faces_robust = face_cascade.detectMultiScale(gray, 1.3, 5)
print(f"Test 2 (No Downsample, scaleFactor 1.3, default minSize): Detected {len(faces_robust)} faces.")

# If face found, test prediction
if len(faces_robust) > 0:
    x, y, w, h = faces_robust[0]
    face_crop = gray[y:y+h, x:x+w]
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_face = _clahe.apply(face_crop)
    resized = cv2.resize(norm_face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    input_data = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    pred = model.predict(input_data)[0]
    emotion = EMOTIONS[np.argmax(pred)]
    confidence = np.max(pred) * 100
    print(f"Prediction: {emotion} ({confidence:.1f}%)")
    print(f"Raw Scores: {dict(zip(EMOTIONS, pred.tolist()))}")
