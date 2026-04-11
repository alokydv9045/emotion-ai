import os
import numpy as np
import tensorflow as tf
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

# Keras 3 Compatibility Patch (Ensuring load works if quantization_config is present)
class LegacyDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

class LegacyConv2D(Conv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

PATCH_OBJECTS = {'Dense': LegacyDense, 'Conv2D': LegacyConv2D, 'LegacyDense': LegacyDense, 'LegacyConv2D': LegacyConv2D}

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.normpath(os.path.join(BASE_DIR, "../dataset/train"))
TEST_DIR = os.path.normpath(os.path.join(BASE_DIR, "../dataset/test"))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "../models/emotion_model.h5"))

# Parameters
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 10 # Fine-tuning requires fewer epochs

# Check if dataset exists
if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Dataset not found at {TRAIN_DIR} or {TEST_DIR}. Please ensure 'dataset/train' and 'dataset/test' exist.")

# ── Hardware Detection ───────────────────────────────────────
print("Checking hardware state...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[TURBO] GPU detected: {len(gpus)} device(s) active.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except: pass
else:
    print("[INFO] No GPU found. Training on CPU (oneDNN optimized).")

# ── Lighting Normalization (CLAHE) ──
def clahe_preprocessing(img):
    img_uint8 = img.astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if img_uint8.shape[-1] == 1:
        norm = clahe.apply(img_uint8[:, :, 0])
        return norm.reshape(img_uint8.shape).astype('float32')
    return img.astype('float32')

# Advanced Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=clahe_preprocessing,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=clahe_preprocessing
)

print(f"Loading data from {TRAIN_DIR}...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# Calculate Class Weights
class_indices = train_generator.class_indices
num_classes = len(class_indices)
y_train = train_generator.classes
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# Pre-calculate steps for Optimizer and fit
steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D, Add, Input
from keras.models import Model

def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    shortcut = x
    
    # Branch
    x = SeparableConv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    
    x = SeparableConv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Residual path logic
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    x = Add()([x, shortcut])
    x = tf.keras.activations.relu(x)
    return x

# ── Model Initialisation ──
if os.path.exists(MODEL_PATH):
    print(f"[INFO] Existing model found. Initialising FINE-TUNING cycle...")
    try:
        model = load_model(MODEL_PATH, custom_objects=PATCH_OBJECTS)
        # Cosine Decay for fine-tuning
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-4, EPOCHS * steps_per_epoch)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}. Starting fresh training.")
        model = None
else:
    model = None

if model is None:
    print("[INFO] Starting fresh High-Fidelity training cycle...")
    img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 2 (Residual)
    x = residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 3 (Residual)
    x = residual_block(x, 256)
    x = MaxPooling2D((2, 2))(x)
    
    # Global Awareness
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=img_input, outputs=output)
    
    # Cosine Decay for fresh training
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, EPOCHS * (train_generator.samples // BATCH_SIZE))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

# Train
steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

print(f"Starting training for {EPOCHS} epochs...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

print(f"Fine-tuning completed. Updated model saved to {MODEL_PATH}")
