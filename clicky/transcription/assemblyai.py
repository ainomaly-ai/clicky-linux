"""AssemblyAI streaming transcription provider.

Connects to AssemblyAI's v3 WebSocket API for real-time transcription.
Mirrors AssemblyAIStreamingTranscriptionProvider.swift from the original macOS app.
"""

import asyncio
import json
import logging
from typing import Callable, Optional, List

import websockets

from clicky.config import config
from clicky.transcription.base import TranscriptionProvider, StreamingTranscriptionSession

logger = logging.getLogger(__name__)

ASSEMBLYAI_WS_URL = "wss://streaming.assemblyai.com/v3/ws"


class AssemblyAISession(StreamingTranscriptionSession):
    """A single AssemblyAI streaming transcription session."""

    def __init__(
        self,
        token: str,
        on_partial_transcript: Callable[[str], None] = None,
        keyterms: Optional[List[str]] = None,
    ):
        self._token = token
        self._on_partial = on_partial_transcript
        self._keyterms = keyterms or []
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._partial_transcript = ""
        self._final_transcript = ""
        self._session_ended = asyncio.Event()
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def final_transcript_fallback_delay_seconds(self) -> float:
        return 2.8

    async def append_audio_buffer(self, pcm_data: bytes):
        """Send PCM16 audio to the AssemblyAI WebSocket."""
        if self._ws and self._ws.open:
            await self._ws.send(pcm_data)

    async def request_final_transcript(self) -> str:
        """Request final transcript and wait for it."""
        if self._ws and self._ws.open:
            # Force endpoint to get final transcript
            await self._ws.send(json.dumps({"type": "ForceEndpoint"}))

            # Wait for final transcript or fallback timeout
            try:
                await asyncio.wait_for(
                    self._session_ended.wait(),
                    timeout=self.final_transcript_fallback_delay_seconds + 1.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for final transcript, using partial")

            # Terminate session
            try:
                await self._ws.send(json.dumps({"type": "Terminate"}))
            except Exception:
                pass

        # Return final or partial
        result = self._final_transcript or self._partial_transcript
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        return result.strip()

    async def cancel(self):
        """Cancel this session."""
        if self._ws and self._ws.open:
            try:
                await self._ws.send(json.dumps({"type": "Terminate"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def connect(self):
        """Open the WebSocket connection and start receiving."""
        params = [
            f"sample_rate=16000",
            f"encoding=pcm_s16le",
            f"format_turns=true",
            f"speech_model=u3_rt_pro",
            f"token={self._token}",
        ]
        if self._keyterms:
            params.append(f"keyterms={','.join(self._keyterms)}")

        url = f"{ASSEMBLYAI_WS_URL}?{'&'.join(params)}"

        self._ws = await websockets.connect(
            url,
            additional_headers={"Authorization": self._token},
            max_size=None,
        )

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        """Receive messages from the WebSocket."""
        try:
            async for message in self._ws:
                data = json.loads(message)

                msg_type = data.get("type")

                if msg_type == "Begin":
                    logger.debug("AssemblyAI session started")

                elif msg_type == "Turn":
                    transcript = data.get("transcript", "")
                    is_end = data.get("end_of_turn", False)
                    is_formatted = data.get("turn_is_formatted", False)

                    if is_end and is_formatted:
                        # Final formatted transcript for this turn
                        self._final_transcript = transcript
                        self._session_ended.set()
                    else:
                        # Partial transcript
                        self._partial_transcript = transcript
                        if self._on_partial:
                            self._on_partial(transcript)

                elif msg_type == "Error":
                    error_msg = data.get("error", "Unknown error")
                    logger.error(f"AssemblyAI error: {error_msg}")
                    self._session_ended.set()

        except websockets.exceptions.ConnectionClosed:
            logger.debug("AssemblyAI WebSocket closed")
        except Exception as e:
            logger.error(f"AssemblyAI receive error: {e}")
        finally:
            self._session_ended.set()


class AssemblyAITranscriptionProvider(TranscriptionProvider):
    """AssemblyAI streaming transcription provider."""

    def __init__(self):
        self._api_key = config.ASSEMBLYAI_API_KEY

    @property
    def display_name(self) -> str:
        return "AssemblyAI"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def _fetch_token(self) -> str:
        """Fetch a temporary streaming token from the proxy or directly."""
        import httpx

        if config.USE_PROXY:
            url = config.transcribe_token_url
            headers = {}
        else:
            url = "https://streaming.assemblyai.com/v3/token"
            headers = {"authorization": self._api_key}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json={"expires_in": 480},
            )
            response.raise_for_status()
            data = response.json()
            return data["token"]

    async def start_streaming_session(
        self,
        on_partial_transcript: Callable[[str], None] = None,
        keyterms: Optional[List[str]] = None,
    ) -> AssemblyAISession:
        """Start a new AssemblyAI streaming session."""
        token = await self._fetch_token()
        session = AssemblyAISession(
            token=token,
            on_partial_transcript=on_partial_transcript,
            keyterms=keyterms,
        )
        await session.connect()
        return session