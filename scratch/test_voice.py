import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

def test_speak():
    url = "http://localhost:8000/speak"
    payload = {
        "text": "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?",
        "emotion": "Happy"
    }
    
    try:
        print(f"Testing {url}...")
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            audio_base64 = data.get("audio", "")
            if audio_base64:
                print(f"Success! Audio received (Size: {len(audio_base64)} chars)")
                # Save for manual verification if needed
                with open("test_audio.mp3", "wb") as f:
                    f.write(base64.b64decode(audio_base64))
                print("Audio saved to test_audio.mp3")
            else:
                print("Error: No audio data in response")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    test_speak()
