import requests
import base64
import time
import concurrent.futures
import os

API_URL = "http://localhost:8000/predict"
TEST_IMG = os.path.join(os.path.dirname(__file__), "dataset", "test", "happy", "PublicTest_93296642.jpg")
TOTAL_REQUESTS = 200
CONCURRENT_USERS = 5

def encode_image(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def send_request(img_b64):
    try:
        start = time.time()
        res = requests.post(API_URL, json={"image": img_b64}, timeout=5)
        latency = (time.time() - start) * 1000
        return res.status_code == 200, latency
    except Exception as e:
        return False, 0

def run_stress_test():
    print(f"INITIATING NEURAL STRESS TEST: {TOTAL_REQUESTS} requests...")
    
    if not os.path.exists(TEST_IMG):
        print(f"ERROR: TEST IMAGE MISSING at {TEST_IMG}")
        return

    img_b64 = encode_image(TEST_IMG)
    latencies = []
    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = [executor.submit(send_request, img_b64) for _ in range(TOTAL_REQUESTS)]
        for future in concurrent.futures.as_completed(futures):
            success, latency = future.result()
            if success:
                success_count += 1
                latencies.append(latency)
            else:
                fail_count += 1

    avg_lat = sum(latencies)/len(latencies) if latencies else 0
    print("\n" + "="*40)
    print("STABILITY AUDIT RESULTS")
    print("="*40)
    print(f"SUCCESS: {success_count}/{TOTAL_REQUESTS}")
    print(f"FAILURE: {fail_count}")
    print(f"AVG LATENCY: {avg_lat:.2f}ms")
    print(f"STATUS: {'PASSED' if fail_count == 0 else 'STABILITY WARNING'}")
    print("="*40)

if __name__ == "__main__":
    run_stress_test()
