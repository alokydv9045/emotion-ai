import sys
import os
import cv2
import numpy as np
import tensorflow as tf

# Add legacy to path
sys.path.append(os.getcwd())

# 1. Load Legacy Predictor
print("Loading Legacy Engine...")
try:
    from legacy.src.predictor import build_precision_model as build_legacy
    from legacy.src.config import MODEL_PATH as LEG_MODEL_PATH
    legacy_model = build_legacy()
    legacy_model.load_weights(LEG_MODEL_PATH, by_name=True)
    print("Legacy Engine Ready.")
except Exception as e:
    print(f"Legacy Load Failed: {e}")
    legacy_model = None

# 2. My Build
print("\nLoading Current Engine (main.py logic)...")
def build_current():
    IMG_SIZE = 48
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
    outputs = tf.keras.layers.Dense(7, activation="softmax", name="dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

current_model = build_current()
current_model.load_weights("backend/models/emotion_model.h5", by_name=True)
print("Current Engine Ready.")

# 3. Test Image
test_file = "dataset/test/happy/PrivateTest_10077120.jpg"
img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
img_norm = cv2.resize(img, (48, 48)).astype("float32") / 255.0
data = img_norm.reshape(1, 48, 48, 1)

print(f"\nComparing prediction for: {test_file}")
if legacy_model:
    l_pred = legacy_model.predict(data, verbose=0)[0]
    print(f"Legacy Index Map: {np.argmax(l_pred)} | Top Score: {np.max(l_pred):.4f}")
    print(f"Legacy Raw: {l_pred}")

c_pred = current_model.predict(data, verbose=0)[0]
print(f"Current Index Map: {np.argmax(c_pred)} | Top Score: {np.max(c_pred):.4f}")
print(f"Current Raw: {c_pred}")
