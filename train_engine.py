import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, Add, 
    GlobalAveragePooling2D, Dropout, Dense, Input, MaxPooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import os
import cv2
import numpy as np
from sklearn.utils import class_weight

tf.keras.backend.set_image_data_format('channels_last')

# CONFIG
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 100
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
DATASET_PATH = "dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
TEST_DIR = os.path.join(DATASET_PATH, "test")
MODEL_SAVE_PATH = "backend/models/emotion_model_v5"
BEST_MODEL_PATH = "backend/models/emotion_model_best"

# ── SHARED PREPROCESSING ──
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def synchronized_preprocessor(img):
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    img_uint8 = (img).astype('uint8')
    img_clahe = clahe.apply(img_uint8)
    img_final = img_clahe.astype('float32') / 255.0
    return np.expand_dims(img_final, axis=-1)

# ── APEX-V5 RESIDUAL ARCHITECTURE ──
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # Path 1
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_apex_v5_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input_layer")
    
    # Stem
    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 512, stride=2)
    
    # Global Output
    x = GlobalAveragePooling2D(name="apex_gap")(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5, name="apex_dropout")(x)
    outputs = Dense(len(EMOTIONS), activation="softmax", name="apex_emotion_softmax")(x)
    
    return Model(inputs=inputs, outputs=outputs)

def train():
    print("--- Apex-V5 Residual Realignment Booting ---")
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=synchronized_preprocessor,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=synchronized_preprocessor
    )
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Class Weights
    labels = train_generator.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
    
    # Build
    model = build_apex_v5_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Stable start
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(BEST_MODEL_PATH + ".h5", monitor='val_accuracy', save_best_only=True, save_weights_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.000001, verbose=1)
    
    print("Initiating Apex-V5 Training...")
    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"Apex-V5 Sequence Complete. Best model at: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    train()
