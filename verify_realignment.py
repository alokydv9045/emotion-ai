import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from train_engine import build_apex_v5_model, synchronized_preprocessor

def verify_accuracy():
    print("--- Apex Phase 2 Accuracy Audit ---")
    
    # 1. Load Model
    model = build_apex_v5_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    BEST_MODEL = 'backend/models/emotion_model_best.h5'
    if os.path.exists(BEST_MODEL):
        model.load_weights(BEST_MODEL, by_name=True)
        print(f"Loaded weights from {BEST_MODEL}")
    else:
        print("ERROR: Weights missing.")
        return

    # 2. Test Generator
    test_datagen = ImageDataGenerator(preprocessing_function=synchronized_preprocessor)
    test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(96, 96),
        color_mode="grayscale",
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # 3. Evaluate
    loss, acc = model.evaluate(test_generator)
    print(f"\n[FINAL AUDIT RESULT]")
    print(f"Categorical Accuracy: {acc*100:.2f}%")
    print(f"Categorical Loss: {loss:.4f}")
    
    if acc > 0.60:
        print("STABILIZATION TARGET: ACHIEVED")
    else:
        print("STABILIZATION TARGET: FAIL - HYPER-TUNING RECOMMENDED")

if __name__ == "__main__":
    verify_accuracy()
