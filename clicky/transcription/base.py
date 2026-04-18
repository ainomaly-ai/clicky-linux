"""Transcription provider protocol and factory.

Mirrors BuddyTranscriptionProvider.swift from the original macOS app.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, List


class StreamingTranscriptionSession(ABC):
    """A single transcription session (one push-to-talk interaction)."""

    @abstractmethod
    async def append_audio_buffer(self, pcm_data: bytes):
        """Append PCM16 audio data to the ongoing transcription."""
        pass

    @abstractmethod
    async def request_final_transcript(self) -> str:
        """Signal end of audio and get the final transcript."""
        pass

    @abstractmethod
    async def cancel(self):
        """Cancel this transcription session."""
        pass

    @property
    @abstractmethod
    def final_transcript_fallback_delay_seconds(self) -> float:
        """Seconds to wait for final transcript before using the latest partial."""
        pass


class TranscriptionProvider(ABC):
    """Base class for transcription backends."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider has all required API keys/settings."""
        pass

    @abstractmethod
    async def start_streaming_session(
        self,
        on_partial_transcript: Callable[[str], None] = None,
        keyterms: Optional[List[str]] = None,
    ) -> StreamingTranscriptionSession:
        """Start a new streaming transcription session."""
        pass


def get_available_provider() -> Optional[TranscriptionProvider]:
    """Get the best available transcription provider.

    Tries AssemblyAI first, then falls back to local Whisper.
    """
    from clicky.config import config

    # Try AssemblyAI first
    if config.ASSEMBLYAI_API_KEY:
        try:
            from clicky.transcription.assemblyai import AssemblyAITranscriptionProvider
            provider = AssemblyAITranscriptionProvider()
            if provider.is_configured:
                return provider
        except Exception:
            pass

    # Fallback to local Whisper
    try:
        from clicky.transcription.whisper_local import WhisperLocalProvider
        provider = WhisperLocalProvider()
        if provider.is_configured:
            return provider
    except Exception:
        pass

    return None