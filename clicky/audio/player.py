"""Audio playback using sounddevice.

Plays MP3/WAV audio data. Used for TTS output from ElevenLabs.
Mirrors ElevenLabsTTSClient.swift audio playback from the original macOS app.
"""

import io
import logging
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Plays audio data (MP3 or WAV) through the default output device.

    Usage:
        player = AudioPlayer()
        player.play(audio_bytes, on_done=lambda: print("done"))
        player.stop()
    """

    def __init__(self):
        self._playing = False
        self._stream: Optional[sd.OutputStream] = None
        self._stop_event = threading.Event()
        self._play_thread: Optional[threading.Thread] = None

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(
        self,
        audio_data: bytes,
        format: str = "mp3",
        on_done: Callable[[], None] = None,
    ):
        """Play audio data.

        Args:
            audio_data: Raw audio bytes (MP3 or WAV)
            format: Audio format ("mp3" or "wav")
            on_done: Called when playback finishes
        """
        self.stop()  # Stop any current playback

        self._stop_event.clear()
        self._playing = True

        def _play_thread():
            try:
                # Decode audio using pydub
                from pydub import AudioSegment

                audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format)

                # Convert to raw PCM float32
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                samples = samples / 32768.0  # Normalize to -1.0 to 1.0

                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)  # Stereo to mono

                sample_rate = audio.frame_rate

                # Play in chunks so we can be interrupted
                chunk_size = 1024
                total_samples = len(samples)
                position = 0

                with sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                ) as stream:
                    while position < total_samples and not self._stop_event.is_set():
                        end = min(position + chunk_size, total_samples)
                        chunk = samples[position:end].reshape(-1, 1)
                        stream.write(chunk)
                        position = end

            except Exception as e:
                logger.error(f"Audio playback error: {e}")
            finally:
                self._playing = False
                if on_done:
                    on_done()

        self._play_thread = threading.Thread(target=_play_thread, daemon=True)
        self._play_thread.start()

    def stop(self):
        """Stop current playback."""
        if self._playing:
            self._stop_event.set()
            self._playing = False
            if self._play_thread and self._play_thread.is_alive():
                self._play_thread.join(timeout=2.0)
            self._play_thread = None
            logger.debug("Playback stopped")

    def wait_until_done(self, timeout: Optional[float] = None):
        """Block until playback finishes or timeout."""
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=timeout)