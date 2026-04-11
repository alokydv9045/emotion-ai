import base64
import requests
import numpy as np
import cv2
import time

def test_health():
    try:
        resp = requests.get("http://localhost:8000/health")
        print(f"Health Check: {resp.status_code} - {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Health Check Failed: {e}")
        return False

def test_predict():
    # Create a dummy gray image (48x48)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "FACE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    _, buffer = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(buffer).decode()
    
    payload = {"image": f"data:image/jpeg;base64,{b64}"}
    
    try:
        start = time.time()
        resp = requests.post("http://localhost:8000/predict", json=payload)
        latency = (time.time() - start) * 1000
        print(f"Predict Check: {resp.status_code} - Latency: {latency:.1f}ms")
        if resp.status_code == 200:
            print(f"Result: {resp.json()}")
            return True
        else:
            print(f"Error Detail: {resp.text}")
            return False
    except Exception as e:
        print(f"Predict Request Failed: {e}")
        return False

if __name__ == "__main__":
    print("--- STARTING SYSTEM INTEGRATION TEST ---")
    h = test_health()
    if h:
        test_predict()
    else:
        print("Backend is not running. Please start launch_production.bat first.")
