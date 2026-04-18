"""Edge TTS client — free neural voice TTS via Microsoft Azure.

Uses edge-tts library which connects to Microsoft Azure's neural TTS engines
for free. No API key needed. Voices like en-US-AriaNeural, en-GB-SoniaNeural.
"""

import asyncio
import logging
import threading
from typing import Callable, Optional

from clicky.audio.player import AudioPlayer

logger = logging.getLogger(__name__)


class EdgeTTSClient:
    """Text-to-speech via Edge TTS (Microsoft Azure neural voices, free)."""

    DEFAULT_VOICE = "en-GB-SoniaNeural"

    def __init__(self, voice: Optional[str] = None, on_playback_finished: Callable[[], None] = None):
        self._voice = voice or self.DEFAULT_VOICE
        self._player = AudioPlayer()
        self._on_playback_finished = on_playback_finished
        self._speaking = False

    @property
    def is_playing(self) -> bool:
        return self._player.is_playing

    @property
    def tts_url(self) -> str:
        return f"edge-tts://{self._voice}"

    async def speak_text(self, text: str):
        """Send text to Edge TTS and play the resulting audio."""
        if not text.strip():
            logger.debug("Empty text, skipping TTS")
            return

        self._speaking = True
        logger.info(f"TTS [edge/{self._voice}] started: {text[:80]}{'...' if len(text) > 80 else ''}")

        try:
            import edge_tts

            # Fetch audio via edge-tts
            communicate = edge_tts.Communicate(text, self._voice)
            audio_data = b""

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if audio_data:
                self._player.stop()
                self._player.play(
                    audio_data,
                    format="mp3",
                    on_done=self._on_playback_finished,
                )
            else:
                logger.warning("No audio data received from Edge TTS")

        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            self._speaking = False
            if self._on_playback_finished:
                self._on_playback_finished()

    def stop_playback(self):
        """Stop current TTS playback."""
        self._player.stop()