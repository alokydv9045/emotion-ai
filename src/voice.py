"""
voice.py — Triple-Tier Hindi TTS Engine
Priority: ElevenLabs -> gTTS (Google, real Hindi) -> pyttsx3 (offline)
"""
import os
import threading
import tempfile
from src.config import (
    VOICE_RATE, VOICE_VOLUME, VOICE_GENDER,
    VOICE_RESPONSES, ELEVENLABS_API_KEY, USE_ELEVENLABS, DEFAULT_VOICE_ID
)

# ── Tier 1: ElevenLabs ───────────────────────────────────────
try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play as el_play
    _elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if (USE_ELEVENLABS and ELEVENLABS_API_KEY) else None
except Exception:
    _elevenlabs_client = None

# ── Tier 2: gTTS (Google TTS, best Hindi pronunciation) ──────
try:
    from gtts import gTTS as _gTTS
    _gtts_available = True
except ImportError:
    _gtts_available = False

# ── Tier 2b: Audio playback for gTTS MP3 ─────────────────────
def _play_mp3(path: str):
    """Play an MP3 file. Tries pygame, then playsound, then os.startfile."""
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        return
    except Exception:
        pass
    try:
        from playsound import playsound
        playsound(path)
        return
    except Exception:
        pass
    # Last resort — Windows only
    try:
        os.startfile(path)
    except Exception:
        pass

# ── Tier 3: pyttsx3 (offline) ────────────────────────────────
try:
    import pyttsx3
    _pyttsx3_engine = pyttsx3.init()
    _pyttsx3_engine.setProperty('rate', VOICE_RATE)
    _pyttsx3_engine.setProperty('volume', VOICE_VOLUME)
    _voices = _pyttsx3_engine.getProperty('voices')
    for _v in _voices:
        if VOICE_GENDER.lower() in _v.name.lower() or 'india' in _v.name.lower():
            _pyttsx3_engine.setProperty('voice', _v.id)
            break
    _pyttsx3_available = True
except Exception:
    _pyttsx3_available = False
    _pyttsx3_engine = None

_speech_lock = threading.Lock()
_last_emotion = ""


def _speak_task(text: str):
    """Internal: runs TTS with fallback chain. Holds speech lock to avoid overlap."""
    with _speech_lock:
        # Tier 1 — ElevenLabs
        if _elevenlabs_client:
            try:
                audio = _elevenlabs_client.generate(
                    text=text,
                    voice=DEFAULT_VOICE_ID,
                    model="eleven_multilingual_v2"
                )
                el_play(audio)
                return
            except Exception as e:
                print(f"[Voice] ElevenLabs failed: {e}")

        # Tier 2 — gTTS (Google Hindi)
        if _gtts_available:
            try:
                tts = _gTTS(text=text, lang='hi', slow=False)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = tmp.name
                tts.save(tmp_path)
                _play_mp3(tmp_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return
            except Exception as e:
                print(f"[Voice] gTTS failed: {e}")

        # Tier 3 — pyttsx3 (offline)
        if _pyttsx3_available and _pyttsx3_engine:
            try:
                _pyttsx3_engine.say(text)
                _pyttsx3_engine.runAndWait()
            except Exception as e:
                print(f"[Voice] pyttsx3 failed: {e}")


def speak(emotion: str):
    """
    Trigger Hindi voice feedback for an emotion.
    Non-blocking — fires in a daemon thread.
    Only speaks when emotion changes (debounced).
    """
    global _last_emotion
    if emotion == _last_emotion:
        return
    _last_emotion = emotion

    text = VOICE_RESPONSES.get(emotion, "")
    if not text:
        return

    t = threading.Thread(target=_speak_task, args=(text,), daemon=True)
    t.start()


def reset_last_emotion():
    """Call this to force voice to re-speak on next detection (e.g. session restart)."""
    global _last_emotion
    _last_emotion = ""
