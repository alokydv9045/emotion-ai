import tensorflow as tf
import os

MODEL_PATH = "backend/models/emotion_model.h5"

print(f"Inspecting {MODEL_PATH}...")
try:
    # Attempt to load the full model (architecture + weights)
    model = tf.keras.models.load_model(MODEL_PATH)
    print("\n[SUCCESS] Model loaded with architecture.")
    model.summary()
except Exception as e:
    print(f"\n[INFO] Could not load architecture (likely weights-only): {e}")
    
    # If weights only, inspect keys
    import h5py
    with h5py.File(MODEL_PATH, 'r') as f:
        print("\nKeys in H5 file:")
        for key in f.keys():
            print(f"- {key}")
            if hasattr(f[key], 'keys'):
                for subkey in f[key].keys():
                    print(f"  -- {subkey}")
