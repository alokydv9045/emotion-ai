import cv2

import threading
import time

# Load once at module level — prevents reloading on every frame (critical perf fix)
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class ThreadedWebcam:
    """
    High-Performance Webcam Thread.
    Continuously pulls frames from the hardware to eliminate buffer bloat.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame

    def stop(self):
        self.started = False
        if self.cap.isOpened():
            self.cap.release()

def get_faces(frame):
    """
    Convert frame to grayscale and detect faces.
    Optimized: Downsampled detection (320x240) to boost FPS.
    Returns (gray, faces_original_resolution).
    """
    if frame is None:
        return None, []
    
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Downsample for detection
    h, w = gray.shape
    scale = 2.0
    small_gray = cv2.resize(gray, (int(w/scale), int(h/scale)))
    
    # 3. Detect
    faces_small = _face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # 4. Scale up
    faces = []
    for (x, y, w_f, h_f) in faces_small:
        faces.append((int(x*scale), int(y*scale), int(w_f*scale), int(h_f*scale)))
        
    return gray, faces
