"""ElevenLabs TTS client.

Sends text to ElevenLabs API and plays back audio.
Mirrors ElevenLabsTTSClient.swift from the original macOS app.
"""

import logging
from typing import Callable, Optional

import httpx

from clicky.config import config
from clicky.audio.player import AudioPlayer

logger = logging.getLogger(__name__)


class ElevenLabsTTSClient:
    """Text-to-speech via ElevenLabs API."""

    def __init__(self, on_playback_finished: Callable[[], None] = None):
        self._api_key = config.ELEVENLABS_API_KEY
        self._voice_id = config.ELEVENLABS_VOICE_ID
        self._player = AudioPlayer()
        self._on_playback_finished = on_playback_finished

    @property
    def is_playing(self) -> bool:
        return self._player.is_playing

    @property
    def tts_url(self) -> str:
        return config.tts_url

    async def speak_text(self, text: str):
        """Send text to ElevenLabs TTS and play the resulting audio.

        Args:
            text: Text to speak (should be stripped of [POINT] tags)
        """
        if not text.strip():
            logger.debug("Empty text, skipping TTS")
            return

        logger.info(f"TTS [elevenlabs] started: {text[:80]}{'...' if len(text) > 80 else ''}")

        if not self._api_key and not config.USE_PROXY:
            logger.warning("ElevenLabs API key not configured, skipping TTS")
            return

        try:
            # Fetch audio from ElevenLabs
            audio_data = await self._fetch_audio(text)

            if audio_data:
                # Stop any current playback
                self._player.stop()

                # Play in a thread (non-blocking)
                self._player.play(
                    audio_data,
                    format="mp3",
                    on_done=self._on_playback_finished,
                )
            else:
                logger.warning("No audio data received from ElevenLabs")

        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Fall back to system TTS
            try:
                from clicky.tts.fallback import speak_fallback
                speak_fallback(text)
            except Exception as fallback_err:
                logger.error(f"Fallback TTS also failed: {fallback_err}")

    async def _fetch_audio(self, text: str) -> Optional[bytes]:
        """Fetch TTS audio from ElevenLabs API or proxy."""
        headers = {"content-type": "application/json"}

        if not config.USE_PROXY:
            headers["xi-api-key"] = self._api_key

        body = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
        }

        if config.USE_PROXY:
            body["voice_id"] = self._voice_id

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                response = await client.post(
                    self.tts_url,
                    json=body,
                    headers=headers,
                )

                if response.status_code == 200:
                    return response.content
                else:
                    logger.error(f"ElevenLabs API error: {response.status_code} {response.text}")
                    return None

        except Exception as e:
            logger.error(f"ElevenLabs request failed: {e}")
            return None

    def stop_playback(self):
        """Stop current TTS playback."""
        self._player.stop()