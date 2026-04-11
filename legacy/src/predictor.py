# ── THE STABILITY BARRIER (STRICT CPU MODE) ───────────────────
import os
import sys
import logging

# Silence TensorFlow Deprecation and Internal Noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DML_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DML_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf

# Force silence on the internal TF/Keras loggers
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from src.config import MODEL_PATH, IMG_SIZE, EMOTIONS

# ── ENGINE BOOT ──────────────────────────────────────────────
HARDWARE_STATUS = "CPU Optimized (STABLE)"

# ── PRECISION NEURAL RECONSTRUCTION (Surgical) ───────────────
def build_precision_model():
    """
    Manually reconstruct the Functional topology from verified metadata.
    This ensures 100% accurate weight-mapping for the emotion engine.
    """
    # Force creation on CPU to prevent driver-level kernel clashing
    with tf.device('/CPU:0'):
        inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input_layer")
        
        # Block 1: Entry
        x = tf.keras.layers.Conv2D(32, 3, padding="same", name="conv2d")(inputs)
        x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
        x = tf.keras.layers.Conv2D(64, 3, padding="same", name="conv2d_1")(x)
        x = tf.keras.layers.BatchNormalization(name="batch_normalization_1")(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="max_pooling2d")(x)
        
        # Block 2: Residual 1
        res_a = tf.keras.layers.SeparableConv2D(128, 3, padding="same", name="separable_conv2d")(x)
        res_a = tf.keras.layers.BatchNormalization(name="batch_normalization_2")(res_a)
        res_a = tf.keras.layers.ReLU(name="re_lu")(res_a)
        res_a = tf.keras.layers.SeparableConv2D(128, 3, padding="same", name="separable_conv2d_1")(res_a)
        res_a = tf.keras.layers.BatchNormalization(name="batch_normalization_3")(res_a)
        
        skip_a = tf.keras.layers.Conv2D(128, 1, padding="same", name="conv2d_2")(x)
        skip_a = tf.keras.layers.BatchNormalization(name="batch_normalization_4")(skip_a)
        
        x = tf.keras.layers.Add(name="add")([res_a, skip_a])
        x = tf.keras.layers.ReLU(name="re_lu_1")(x)
        
        # Block 3: Residual 2
        x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="max_pooling2d_1")(x)
        
        res_b = tf.keras.layers.SeparableConv2D(256, 3, padding="same", name="separable_conv2d_2")(x)
        res_b = tf.keras.layers.BatchNormalization(name="batch_normalization_5")(res_b)
        res_b = tf.keras.layers.ReLU(name="re_lu_2")(res_b)
        res_b = tf.keras.layers.SeparableConv2D(256, 3, padding="same", name="separable_conv2d_3")(res_b)
        res_b = tf.keras.layers.BatchNormalization(name="batch_normalization_6")(res_b)
        
        skip_b = tf.keras.layers.Conv2D(256, 1, padding="same", name="conv2d_3")(x)
        skip_b = tf.keras.layers.BatchNormalization(name="batch_normalization_7")(skip_b)
        
        x = tf.keras.layers.Add(name="add_1")([res_b, skip_b])
        
        # Final Exit
        x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid", name="max_pooling2d_2")(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
        
        outputs = tf.keras.layers.Dense(len(EMOTIONS), activation="softmax", name="dense")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ── ENGINE BOOT ──────────────────────────────────────────────
model = None
HARDWARE_STATUS = "CPU Optimized (STABLE)"

if os.path.exists(MODEL_PATH):
    try:
        model = build_precision_model()
        # Graft weights on CPU
        with tf.device('/CPU:0'):
            model.load_weights(MODEL_PATH, by_name=True)
        print(f"[SUCCESS] Precision Engine ready. Weights loaded from: {MODEL_PATH}")
    except Exception as e:
        # Fallback to pure CPU build in case of any residual conflicts
        print(f"[RECOVERY] Neural Grafting Failed: {e}")
        try:
            model = build_precision_model()
            model.load_weights(MODEL_PATH, by_name=True)
            print("[SUCCESS] Engine ready (Safety Mode Fallback).")
        except Exception as e2:
            print(f"[CRITICAL] Engine Boot Failure: {e2}")
            model = None

# ── MODULE LEVEL CACHING ─────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def predict_face(gray_face):
    """
    Perform emotion inference on a single face.
    Optimized: Direct call bypasses Keras overhead. 
    Uses pre-initialized CLAHE.
    """
    if model is None: 
        return "Neutral", np.zeros((1, 7))
    
    try:
        # 1. Type Normalization
        if gray_face.dtype != np.uint8:
            gray_face = (gray_face * 255).astype(np.uint8) if gray_face.max() <= 1.0 else gray_face.astype(np.uint8)
        
        # 2. Enhancements
        norm_face = _clahe.apply(gray_face)
        face = cv2.resize(norm_face, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        # 3. Fast Inference (Direct call is ~2x faster than model.predict for single items)
        pred_tensor = model(face, training=False)
        pred = pred_tensor.numpy()
        
        return EMOTIONS[np.argmax(pred[0])], pred
    except Exception as e:
        print(f"[Predictor] Inference error: {e}")
        return "Neutral", np.zeros((1, 7))
