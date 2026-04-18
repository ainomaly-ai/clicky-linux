"""Fallback TTS using pyttsx3 / espeak for offline text-to-speech.

Used when ElevenLabs API key is not configured or as a fallback.
Mirrors the NSSpeechSynthesizer fallback in the original macOS app.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    """Lazy-initialize the pyttsx3 engine."""
    global _engine
    if _engine is None:
        try:
            import pyttsx3
            _engine = pyttsx3.init()
            _engine.setProperty("rate", 180)
            _engine.setProperty("volume", 1.0)
            # Try to use a good voice
            voices = _engine.getProperty("voices")
            for voice in voices:
                if "english" in voice.name.lower():
                    _engine.setProperty("voice", voice.id)
                    break
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise
    return _engine


def speak_fallback(text: str):
    """Speak text using local TTS (pyttsx3/espeak).

    This runs synchronously in the calling thread.
    For async usage, call this from a thread.
    """
    if not text.strip():
        return

    try:
        with _engine_lock:
            engine = _get_engine()
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        logger.error(f"Fallback TTS error: {e}")
        # Last resort: espeak command
        try:
            import subprocess
            subprocess.run(["espeak", text], check=False, timeout=10)
        except Exception:
            logger.error("espeak also failed, no TTS available")


def speak_fallback_async(text: str):
    """Speak text in a background thread (non-blocking)."""
    thread = threading.Thread(target=speak_fallback, args=(text,), daemon=True)
    thread.start()
    return thread