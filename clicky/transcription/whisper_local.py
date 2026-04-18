"""Local Whisper transcription provider for offline fallback.

Uses faster-whisper for local, on-device transcription.
Mirrors AppleSpeechTranscriptionProvider.swift as the local fallback.
"""

import logging
import io
import tempfile
from typing import Callable, Optional, List

from clicky.config import config
from clicky.transcription.base import TranscriptionProvider, StreamingTranscriptionSession
from clicky.audio.converter import pcm16_to_wav

logger = logging.getLogger(__name__)


class WhisperLocalSession(StreamingTranscriptionSession):
    """A local Whisper transcription session.

    Buffers audio during recording, then transcribes on request_final_transcript.
    This is not truly streaming - it buffers and transcribes at the end.
    """

    def __init__(
        self,
        model_name: str = "tiny",
        on_partial_transcript: Callable[[str], None] = None,
    ):
        self._model_name = model_name
        self._on_partial = on_partial_transcript
        self._audio_buffers: List[bytes] = []
        self._model = None

    @property
    def final_transcript_fallback_delay_seconds(self) -> float:
        return 0.0  # No fallback needed, we transcribe on demand

    async def append_audio_buffer(self, pcm_data: bytes):
        """Buffer audio data for later transcription."""
        self._audio_buffers.append(pcm_data)

    async def request_final_transcript(self) -> str:
        """Transcribe the buffered audio."""
        if not self._audio_buffers:
            return ""

        try:
            from faster_whisper import WhisperModel

            # Combine all audio buffers
            all_pcm = b"".join(self._audio_buffers)
            wav_data = pcm16_to_wav(all_pcm, sample_rate=16000, channels=1)

            # Write to temp file for faster-whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_data)
                temp_path = f.name

            # Load model and transcribe
            model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
            segments, info = model.transcribe(temp_path, language="en")

            transcript = " ".join(segment.text.strip() for segment in segments)
            logger.info(f"Whisper transcription: {transcript}")

            # Cleanup
            import os
            os.unlink(temp_path)

            return transcript.strip()

        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            return ""
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ""

    async def cancel(self):
        """Cancel - just discard buffered audio."""
        self._audio_buffers.clear()


class WhisperLocalProvider(TranscriptionProvider):
    """Local Whisper transcription provider (offline fallback)."""

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or config.WHISPER_MODEL

    @property
    def display_name(self) -> str:
        return f"Whisper Local ({self._model_name})"

    @property
    def is_configured(self) -> bool:
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    async def start_streaming_session(
        self,
        on_partial_transcript: Callable[[str], None] = None,
        keyterms: Optional[List[str]] = None,
    ) -> WhisperLocalSession:
        """Start a local Whisper session (buffers audio for batch transcription)."""
        return WhisperLocalSession(
            model_name=self._model_name,
            on_partial_transcript=on_partial_transcript,
        )