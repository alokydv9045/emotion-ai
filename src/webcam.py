import cv2

# Load once at module level — prevents reloading on every frame (critical perf fix)
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_faces(frame):
    """Convert frame to grayscale and detect faces. Returns (gray, faces)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return gray, faces
