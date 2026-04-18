"""pyttsx3 TTS client — offline text-to-speech using system TTS engine.

Uses espeak (Linux), SAPI5 (Windows), or NSSpeechSynthesizer (macOS).
No API key needed.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    """Lazy-init pyttsx3 engine."""
    global _engine
    if _engine is None:
        try:
            import pyttsx3
            _engine = pyttsx3.init()
            _engine.setProperty("rate", 180)
            _engine.setProperty("volume", 1.0)
            voices = _engine.getProperty("voices")
            for voice in voices:
                if "english" in voice.name.lower():
                    _engine.setProperty("voice", voice.id)
                    break
            logger.info(f"pyttsx3 initialized, {len(voices)} voices available")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise
    return _engine


class Pyttsx3TTSClient:
    """Offline TTS via pyttsx3 / system TTS engine."""

    def __init__(self, on_playback_finished=None):
        self._on_playback_finished = on_playback_finished
        self._speaking = False
        self._thread: threading.Thread = None

    @property
    def is_playing(self) -> bool:
        return self._speaking

    @property
    def tts_url(self) -> str:
        return ""  # Not applicable for local TTS

    async def speak_text(self, text: str):
        """Speak text using system TTS (non-blocking)."""
        if not text.strip():
            logger.debug("Empty text, skipping TTS")
            return

        self._speaking = True
        logger.info(f"TTS [pyttsx3] started: {text[:80]}{'...' if len(text) > 80 else ''}")

        def _speak():
            try:
                with _engine_lock:
                    engine = _get_engine()
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                logger.error(f"pyttsx3 TTS error: {e}")
                # Last resort: espeak command
                try:
                    import subprocess
                    subprocess.run(["espeak", text], check=False, timeout=10)
                except Exception:
                    logger.error("espeak also failed")
            finally:
                self._speaking = False
                if self._on_playback_finished:
                    self._on_playback_finished()

        self._thread = threading.Thread(target=_speak, daemon=True)
        self._thread.start()

    def stop_playback(self):
        """Stop current TTS playback."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            self._speaking = False
        except Exception:
            pass