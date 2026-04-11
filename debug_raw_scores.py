import cv2
import numpy as np
import tensorflow as tf
import os

IMG_SIZE = 48
MODEL_PATH = "backend/models/emotion_model.h5"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def build():
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
    outputs = tf.keras.layers.Dense(7, activation="softmax", name="dense")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = build()
model.load_weights(MODEL_PATH, by_name=True)

test_file = "dataset/test/happy/PrivateTest_10077120.jpg"
img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48)).astype("float32") / 255.0
input_data = img.reshape(1, 48, 48, 1)

pred = model.predict(input_data)[0]
print(f"File: {test_file}")
for i, score in enumerate(pred):
    print(f"Index {i}: {score:.4f}")

max_idx = np.argmax(pred)
print(f"Top Index: {max_idx}")
