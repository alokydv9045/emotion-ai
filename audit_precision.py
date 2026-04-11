import os
import cv2
import numpy as np
import tensorflow as tf
import requests
import base64
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000/predict"
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset", "test")
CATEGORIES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SAMPLES_PER_CAT = 50 # Comprehensive but fast

def encode_image(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def test_single_image(args):
    img_path, true_label = args
    img_b64 = encode_image(img_path)
    try:
        res = requests.post(API_URL, json={"image": img_b64}, timeout=5)
        if res.status_code == 200:
            pred_label = res.json().get("emotion", "Unknown").lower()
            return true_label == pred_label, true_label, pred_label
    except:
        pass
    return False, true_label, "error"

def run_precision_audit():
    print(f"INITIATING CATEGORICAL PRECISION AUDIT...")
    tasks = []
    
    for cat in CATEGORIES:
        cat_path = os.path.join(TEST_DATA_DIR, cat)
        if not os.path.exists(cat_path): continue
        
        files = [f for f in os.listdir(cat_path) if f.endswith('.jpg')][:SAMPLES_PER_CAT]
        for f in files:
            tasks.append((os.path.join(cat_path, f), cat))

    print(f"Benchmarking {len(tasks)} images across {len(CATEGORIES)} categories...")
    
    results = {"total": 0, "correct": 0, "by_category": {cat: {"total": 0, "correct": 0} for cat in CATEGORIES}}

    with ThreadPoolExecutor(max_workers=10) as executor:
        for success, true_label, pred_label in executor.map(test_single_image, tasks):
            results["total"] += 1
            results["by_category"][true_label]["total"] += 1
            if success:
                results["correct"] += 1
                results["by_category"][true_label]["correct"] += 1

    print("\n" + "="*40)
    print("PRECISION AUDIT RESULTS")
    print("="*40)
    total_acc = (results["correct"] / results["total"] * 100) if results["total"] else 0
    print(f"OVERALL ACCURACY: {total_acc:.2f}%")
    print("-" * 40)
    for cat in CATEGORIES:
        stats = results["by_category"][cat]
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0
        print(f"{cat.upper():<10}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
    print("="*40)

if __name__ == "__main__":
    run_precision_audit()
