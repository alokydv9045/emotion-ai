# 📑 ANNEXURES / APPENDIX

This section provides supplementary technical data, architectural diagrams, and empirical results supporting the **Emotion Detection System** documentation.

---

## Appendix A – Dataset Attributes (FER2013)

The system is trained on the **Facial Expression Recognition 2013 (FER2013)** dataset, which was originally introduced for the ICML 2013 challenges.

| Attribute | Specification |
| :--- | :--- |
| **Total Images** | 35,887 |
| **Resolution** | 48 x 48 Pixels (Upscaled to 96x96 for Apex-v5) |
| **Color Space** | Grayscale (1-Channel) |
| **Emotion Categories** | Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral |
| **Data Split** | Training (28,709), Public Test (3,589), Private Test (3,589) |
| **Source** | Kaggle / Pierre-Luc Carrier & Aaron Courville |

---

## Appendix B – Data Augmentation Logic

To improve model generalization and prevent overfitting, the following real-time augmentations are applied during the training phase via the `ImageDataGenerator` or `tf.data` pipeline.

- **Rotation Range**: ±15° (Handles slight head tilts).
- **Width/Height Shift**: 10% (Handles off-center facial positioning).
- **Shear Range**: 0.1 (Simulates perspective distortion).
- **Zoom Range**: 0.1 (Simulates varying distances from the camera).
- **Horizontal Flip**: Enabled (Crucial for facial symmetry invariance).
- **Fill Mode**: 'nearest' (Handles pixel gaps after transformation).

---

## Appendix C – Neural Apex-v5 Stem Architecture

The **Apex-v5** stem is designed for aggressive feature extraction in the initial layers to capture macro-level facial landmarks.

```python
# Stem Architecture Snippet (TensorFlow/Keras)
def build_apex_v5_stem(IMG_SIZE=96):
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Large receptive field for landmark detection
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    # Followed by Residual Stages...
    return x
```

---

## Appendix D – Model Training Graphs

During the training of **Phase 2 Weights**, the model exhibited steady convergence.

- **Loss Curve**: Showed a sharp decline in the first 20 epochs, followed by a plateau. Label smoothing (0.1) prevented the loss from reaching absolute zero, maintaining a healthy margin for generalization.
- **Accuracy Curve**: 
    - **Training Accuracy**: ~74.2%
    - **Validation Accuracy**: ~68.1%
- **Convergence**: Early stopping triggered at Epoch 82 to prevent overfitting as validation loss began to oscillate.

---

## Appendix E – Confusion Matrix

Typical performance distribution for the **Apex-v5 Engine** on the private test set:

| | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Angry** | **62%** | 2% | 12% | 3% | 10% | 2% | 9% |
| **Disgust** | 15% | **76%** | 4% | 1% | 2% | 1% | 1% |
| **Fear** | 10% | 1% | **54%** | 2% | 14% | 12% | 7% |
| **Happy** | 1% | 0% | 2% | **89%** | 2% | 3% | 3% |
| **Sad** | 8% | 1% | 11% | 2% | **58%** | 1% | 19% |
| **Surprise** | 2% | 0% | 14% | 3% | 1% | **78%** | 2% |
| **Neutral** | 5% | 0% | 4% | 3% | 16% | 2% | **70%** |

---

## Appendix F – Dashboard Screenshots

The **Cyber Dashboard** utilizes a premium **Glassmorphism** design system with a high-contrast "Cyber Cyan" color palette.

- **Live Engage Mode**: Features a real-time video stream with a dynamic SVG reticle that centers on the detected face.
- **Neural Bars**: Seven animated progress bars representing real-time confidence scores for each emotion.
- **System Logs**: A translucent overlay in the bottom-right corner displaying real-time API latency and hardware status.

---

## Appendix G – API Endpoint Documentation

The system exposes a RESTful API via FastAPI on port 8000.

### 1. Prediction Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "image": "data:image/jpeg;base64,...",
    "stream": true,
    "session_id": "user_123"
  }
  ```
- **Description**: Processes a single frame and returns emotion probabilities.

### 2. Voice Synthesis Endpoint
- **URL**: `/speak`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "text": "Hello, how are you?",
    "emotion": "Happy"
  }
  ```
- **Description**: Returns a Base64 encoded MP3 audio stream using ElevenLabs Flash v2.5.

### 3. System Health
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Returns the current status of the neural engine and connected hardware.

---
*Created by Team CodeCrafters - 2026*
