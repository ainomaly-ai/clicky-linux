"""Push-to-talk audio recorder using sounddevice.

Captures microphone audio as PCM16 mono at 16kHz.
Supports VAD (Voice Activity Detection) mode for auto-listen.
Mirrors BuddyDictationManager.swift from the original macOS app.
"""

import logging
import threading
import time
from typing import Callable, Optional, List

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16


class AudioRecorder:
    """Records microphone audio for push-to-talk.

    Usage:
        recorder = AudioRecorder()
        recorder.start_recording(on_audio_level=lambda level: ...)
        # ... user is speaking ...
        audio_data = recorder.stop_recording()  # Returns PCM16 bytes
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._audio_buffers: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._on_audio_level: Callable[[float], None] = None
        self._current_level = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start_recording(
        self,
        on_audio_level: Callable[[float], None] = None,
    ):
        """Start recording from the microphone.

        Args:
            on_audio_level: Optional callback with audio power level (0.0-1.0)
                            for waveform visualization.
        """
        with self._lock:
            if self._recording:
                logger.warning("Already recording, ignoring start request")
                return

            self._recording = True
            self._audio_buffers = []
            self._on_audio_level = on_audio_level

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=1024,
                    callback=self._audio_callback,
                )
                self._stream.start()
                logger.info("Recording started")
            except Exception as e:
                self._recording = False
                logger.error(f"Failed to start recording: {e}")
                raise

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """sounddevice callback - receives audio chunks."""
        if status:
            logger.warning(f"Audio status: {status}")

        # Convert float32 to int16 PCM
        pcm_data = (indata[:, 0] * 32767).astype(np.int16)
        self._audio_buffers.append(pcm_data.copy())

        # Calculate audio level for waveform
        if self._on_audio_level:
            rms = np.sqrt(np.mean(indata[:, 0] ** 2))
            level = min(rms * 5.0, 1.0)  # Scale to 0-1
            self._on_audio_level(level)

    def stop_recording(self) -> bytes:
        """Stop recording and return PCM16 audio data.

        Returns:
            Raw PCM16 mono bytes at the configured sample rate.
        """
        with self._lock:
            if not self._recording:
                return b""

            self._recording = False

            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            if not self._audio_buffers:
                return b""

            # Concatenate all buffers
            all_audio = np.concatenate(self._audio_buffers)
            self._audio_buffers = []

            logger.info(f"Recording stopped, captured {len(all_audio)} samples ({len(all_audio) / self.sample_rate:.1f}s)")
            return all_audio.tobytes()

    def cancel_recording(self):
        """Cancel recording without returning audio data."""
        with self._lock:
            self._recording = False
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._audio_buffers = []
            logger.info("Recording cancelled")

    def get_current_audio_level(self) -> float:
        """Get the current audio level (0.0-1.0) for waveform display."""
        return self._current_level


class VoiceActivityDetector:
    """Continuously monitors audio for voice activity (VAD).

    When level exceeds threshold -> fires on_voice_start
    When silence persists for timeout_seconds -> fires on_voice_end
    Used for AUTO_LISTEN mode (no hotkey needed).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        threshold: float = 0.15,
        silence_timeout: float = 2.0,
        on_audio_level: Callable[[float], None] = None,
        on_voice_start: Callable[[], None] = None,
        on_voice_end: Callable[[bytes], None] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.threshold = threshold
        self.silence_timeout = silence_timeout
        self.on_audio_level = on_audio_level
        self.on_voice_start = on_voice_start
        self.on_voice_end = on_voice_end

        self._running = False
        self._stream: Optional[sd.InputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_speaking = False
        self._silence_start: Optional[float] = None
        self._audio_buffers: List[np.ndarray] = []

    def start(self):
        """Start VAD monitoring in a background thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._is_speaking = False
            self._silence_start = None
            self._audio_buffers = []

        self._thread = threading.Thread(target=self._run, daemon=True, name="clicky-vad")
        self._thread.start()
        logger.info("VAD started")

    def stop(self):
        """Stop VAD monitoring."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info("VAD stopped")

    def _run(self):
        """VAD monitoring loop running in background thread."""

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"VAD audio status: {status}")
            rms = np.sqrt(np.mean(indata[:, 0] ** 2))
            level = min(rms * 5.0, 1.0)

            if self.on_audio_level:
                self.on_audio_level(level)

            now = time.time()

            if level > self.threshold:
                # Voice detected
                if not self._is_speaking:
                    self._is_speaking = True
                    self._audio_buffers = []
                    self._silence_start = None
                    logger.info(f"VAD: voice start (level={level:.3f})")
                    if self.on_voice_start:
                        self.on_voice_start()
                # Extend audio buffers while speaking
                self._audio_buffers.append(indata[:, 0].copy())
                # Reset silence timer
                self._silence_start = None
            else:
                # Silence
                if self._is_speaking:
                    # Still capturing — record silence buffer too
                    self._audio_buffers.append(indata[:, 0].copy())
                    if self._silence_start is None:
                        self._silence_start = now
                    elif now - self._silence_start >= self.silence_timeout:
                        # Silence timeout — end of utterance
                        logger.info(f"VAD: voice end (silence timeout)")
                        self._is_speaking = False
                        self._silence_start = None
                        audio_data = np.concatenate(self._audio_buffers)
                        self._audio_buffers = []
                        pcm_bytes = (audio_data * 32767).astype(np.int16).tobytes()
                        if self.on_voice_end:
                            self.on_voice_end(pcm_bytes)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=1024,
                callback=callback,
            )
            self._stream.start()
            # Block while running
            while self._running:
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"VAD stream error: {e}")
        finally:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None