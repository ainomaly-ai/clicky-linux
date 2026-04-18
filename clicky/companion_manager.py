"""CompanionManager - Central state machine for Clicky.

Manages the voice interaction flow:
  IDLE -> LISTENING (hotkey pressed) -> PROCESSING (hotkey released) -> RESPONDING (TTS speaking) -> IDLE

Wires together:
  - Hotkey -> AudioRecorder
  - AudioRecorder -> TranscriptionProvider
  - Transcription -> ScreenCapture + ClaudeAPI
  - Claude response -> OverlayManager (pointing) + ElevenLabsTTS (speech)
  - State changes -> TrayIcon + OverlayManager UI updates
"""

import asyncio
import logging
import threading
from enum import Enum
from typing import Callable, Optional, List

from clicky.config import config
from clicky.hotkey import GlobalHotkeyMonitor, ShortcutTransition
from clicky.audio.recorder import AudioRecorder, VoiceActivityDetector
from clicky.audio.converter import pcm16_to_wav
from clicky.transcription.base import get_available_provider, StreamingTranscriptionSession
from clicky.vision.claude_api import ClaudeAPI, parse_pointing_coordinates, strip_pointing_tags, parse_scroll_commands
from clicky.vision.ollama_api import OllamaAPI
from clicky.screen.capture import ScreenCaptureUtility
from clicky.tts.elevenlabs import ElevenLabsTTSClient
from clicky.tts.pyttsx3_client import Pyttsx3TTSClient
from clicky.tts.edge_tts_client import EdgeTTSClient

from pynput import keyboard as pynput_keyboard

logger = logging.getLogger(__name__)


def _execute_scroll(direction: str, count: int | float):
    """Send PageUp/PageDown key presses for scrolling.

    Args:
        direction: "up" or "down"
        count: integer for page counts, or float for fractional page scrolls (0.0-1.0+)
    """
    try:
        kb = pynput_keyboard.Controller()
        if isinstance(count, float):
            # Fractional scroll — press scroll amount proportional to page
            presses = max(1, round(count * 3))  # 0.5 = ~1.5 presses → 2
            for _ in range(presses):
                key = pynput_keyboard.Key.page_down if direction == "down" else pynput_keyboard.Key.page_up
                kb.press(key)
                kb.release(key)
                time.sleep(0.05)
            logger.info(f"Scrolled {direction} {count} ({presses} key presses)")
        else:
            for _ in range(count):
                key = pynput_keyboard.Key.page_down if direction == "down" else pynput_keyboard.Key.page_up
                kb.press(key)
                kb.release(key)
                time.sleep(0.05)
            logger.info(f"Scrolled {direction} {count}x")
    except Exception as e:
        logger.error(f"Scroll failed: {e}")


def _create_tts_backend():
    """Create TTS backend based on TTS_BACKEND config.

    - "edge" -> Edge TTS (free neural voices, no API key) [DEFAULT]
    - "pyttsx3" -> offline system TTS (no API key, robotic quality)
    - "elevenlabs" -> ElevenLabs API (API key needed)
    """
    backend = config.TTS_BACKEND.lower()
    if backend == "edge":
        logger.info(f"Using Edge TTS (voice={config.EDGE_TTS_VOICE})")
        return EdgeTTSClient(voice=config.EDGE_TTS_VOICE, on_playback_finished=None)
    elif backend == "pyttsx3":
        logger.info("Using pyttsx3 (offline TTS)")
        return Pyttsx3TTSClient(on_playback_finished=None)
    logger.info(f"Using ElevenLabs TTS backend")
    return ElevenLabsTTSClient(on_playback_finished=None)


def _create_vision_backend():
    """Create the vision API backend based on configuration.
    
    Routes based on GUIDING_AGENT config:
    - "hermes" -> HermesAgentBackend (with tools, memory, skills)
    - "llm" -> Direct LLM backend (Ollama, llama.cpp, Anthropic, OpenAI)
    Falls back to LLM if Hermes agent binary is not found.
    """
    # Check if we should use Hermes agent
    if config.GUIDING_AGENT.lower() == "hermes":
        try:
            from clicky.vision.hermes_agent_backend import HermesAgentBackend
            backend = HermesAgentBackend()
            logger.info(
                f"Using Hermes agent backend "
                f"(binary: {backend._agent_binary}, skills: {config.HERMES_AGENT_SKILLS})"
            )
            return backend
        except RuntimeError as e:
            logger.warning(f"Hermes agent not available, falling back to LLM: {e}")
        except ImportError:
            logger.warning("Hermes agent backend module not found, using LLM backend")

    # Default: LLM backend
    backend = config.LLM_BACKEND.lower()
    if backend == "ollama":
        logger.info(f"Using Ollama backend: {config.OLLAMA_MODEL} @ {config.OLLAMA_BASE_URL}")
        return OllamaAPI(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
        )
    elif backend == "llamacpp":
        from clicky.vision.ollama_api import OllamaAPI as OpenAICompatAPI
        model = config.LLAMACPP_MODEL or None  # llamacpp may not need model name
        extra_body = {}
        if config.LLAMACPP_DISABLE_THINKING:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        logger.info(f"Using llama.cpp backend @ {config.LLAMACPP_BASE_URL}" + (f" model={model}" if model else "") + (" (thinking disabled)" if extra_body else ""))
        return OpenAICompatAPI(
            base_url=config.LLAMACPP_BASE_URL,
            model=model,
            extra_body=extra_body,
        )
    elif backend == "openai":
        from clicky.vision.openai_api import OpenAIVisionAPI
        logger.info("Using OpenAI backend")
        return OpenAIVisionAPI()
    elif backend == "anthropic":
        logger.info(f"Using Anthropic backend: {config.CLAUDE_MODEL}")
        return ClaudeAPI()
    else:
        logger.warning(f"Unknown LLM_BACKEND '{backend}', falling back to Ollama")
        return OllamaAPI(base_url=config.OLLAMA_BASE_URL, model=config.OLLAMA_MODEL)


class CompanionVoiceState(Enum):
    """Voice interaction states. Used by TrayIcon and OverlayManager."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"


class CompanionManager:
    """Central state machine orchestrating the Clicky voice interaction loop.

    Usage:
        manager = CompanionManager(overlay_manager=overlay, tray_icon=tray)
        manager.initialize(app)  # Sets up hotkey, recorder, etc.
        # ... run PyQt6 event loop ...
        manager.cleanup()
    """

    def __init__(
        self,
        overlay_manager,
        tray_icon=None,
        on_state_change: Callable[[CompanionVoiceState], None] = None,
    ):
        self._overlay = overlay_manager
        self._tray = tray_icon
        self._on_state_change = on_state_change

        self._state = CompanionVoiceState.IDLE
        self._recorder = AudioRecorder(
            sample_rate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
        )
        self._hotkey = GlobalHotkeyMonitor(
            combo=config.HOTKEY_COMBO,
            on_transition=self._on_hotkey_transition,
        )
        self._vision = _create_vision_backend()
        self._screen = ScreenCaptureUtility(
            max_dimension=config.SCREENSHOT_MAX_DIMENSION,
            jpeg_quality=config.SCREENSHOT_JPEG_QUALITY,
        )
        self._tts = _create_tts_backend()
        # Set callback after creation (method reference resolved after self exists)
        self._tts._on_playback_finished = self._on_tts_finished

        # Transcription session (created fresh each interaction)
        self._transcription_session: Optional[StreamingTranscriptionSession] = None
        self._transcription_provider = None

        # Async event loop running in a background thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

        # Buffer of PCM16 audio chunks collected during recording
        self._audio_chunks: List[bytes] = []
        self._audio_level_timer = None

        # Conversation history clearing timer
        self._idle_timer = None

        # VAD (Voice Activity Detection) for AUTO_LISTEN mode
        self._vad: Optional[VoiceActivityDetector] = None
        if config.AUTO_LISTEN:
            self._vad = VoiceActivityDetector(
                sample_rate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                threshold=config.AUTO_LISTEN_THRESHOLD,
                silence_timeout=config.AUTO_LISTEN_TIMEOUT,
                on_audio_level=self._on_audio_level,
                on_voice_start=self._on_vad_voice_start,
                on_voice_end=self._on_vad_voice_end,
            )

    @property
    def state(self) -> CompanionVoiceState:
        return self._state

    def _set_state(self, new_state: CompanionVoiceState):
        """Update state and notify all observers."""
        if self._state == new_state:
            return
        old = self._state
        self._state = new_state
        logger.info(f"State: {old.value} -> {new_state.value}")

        # Update tray icon
        if self._tray:
            self._tray.update_state(new_state)

        # Update overlay
        if self._overlay:
            if new_state == CompanionVoiceState.IDLE:
                self._overlay.hide_cursor()
                self._overlay.set_processing(False)
                self._overlay.set_response_text("")
            elif new_state == CompanionVoiceState.LISTENING:
                self._overlay.show_cursor()
                self._overlay.set_processing(False)
                self._overlay.set_response_text("")
            elif new_state == CompanionVoiceState.PROCESSING:
                self._overlay.show_cursor()
                self._overlay.set_processing(True)
            elif new_state == CompanionVoiceState.RESPONDING:
                self._overlay.show_cursor()
                self._overlay.set_processing(False)

        # Callback
        if self._on_state_change:
            self._on_state_change(new_state)

    def initialize(self, app):
        """Start the async event loop and hotkey monitor.

        Args:
            app: QApplication instance (for overlay initialization)
        """
        # Start background asyncio event loop
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="clicky-async",
        )
        self._loop_thread.start()

        # Start hotkey monitor
        self._hotkey.start()
        logger.info(f"CompanionManager initialized. Hotkey: {config.HOTKEY_COMBO}")

        # Start VAD for auto-listen mode
        if self._vad:
            self._vad.start()
            logger.info(f"VAD auto-listen enabled (threshold={config.AUTO_LISTEN_THRESHOLD}, timeout={config.AUTO_LISTEN_TIMEOUT}s)")

    def _run_loop(self):
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def cleanup(self):
        """Shut down all components."""
        self._hotkey.stop()
        self._screen.cleanup()
        self._tts.stop_playback()
        if self._vad:
            self._vad.stop()

        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

        logger.info("CompanionManager cleaned up")

    # ── Hotkey Handler ──────────────────────────────────────────────────

    def _on_hotkey_transition(self, transition: ShortcutTransition):
        """Called from the hotkey listener thread.

        Handles both push-to-talk and interrupt:
        - PRESSED in IDLE: start recording
        - PRESSED in LISTENING/PROCESSING/RESPONDING: interrupt current operation
        - RELEASE in LISTENING: stop recording and process
        """
        if transition == ShortcutTransition.PRESSED:
            if self._state == CompanionVoiceState.IDLE:
                self._start_listening()
            else:
                # Interrupt: cancel current operation and return to IDLE
                # (don't call _start_listening here — hotkey release will do it)
                self._interrupt_current_operation()
        elif transition == ShortcutTransition.RELEASED:
            if self._state == CompanionVoiceState.LISTENING:
                self._stop_listening()

    def _start_listening(self):
        """Hotkey pressed: start recording audio and transcription."""
        if self._state != CompanionVoiceState.IDLE:
            logger.debug(f"Ignoring hotkey press in state {self._state.value}")
            return

        self._set_state(CompanionVoiceState.LISTENING)
        self._audio_chunks.clear()

        # Get transcription provider
        self._transcription_provider = get_available_provider()
        if not self._transcription_provider:
            logger.warning("No transcription provider available!")
            self._set_state(CompanionVoiceState.IDLE)
            return

        # Start recording with audio level callback
        self._recorder.start_recording(
            on_audio_level=self._on_audio_level
        )

        # Start transcription session in async loop
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._start_transcription(), self._loop
            )

    def _stop_listening(self):
        """Hotkey released: stop recording, finalize transcription, process."""
        if self._state != CompanionVoiceState.LISTENING:
            return

        self._set_state(CompanionVoiceState.PROCESSING)
        self._overlay.set_audio_level(0.0)

        # Stop recording and get PCM16 audio
        pcm_data = self._recorder.stop_recording()

        if not pcm_data:
            logger.warning("No audio recorded")
            self._set_state(CompanionVoiceState.IDLE)
            return

        # Store the full audio for final transcript
        self._audio_chunks.append(pcm_data)

        # Finalize transcription and process in async loop
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._process_interaction(pcm_data), self._loop
            )

    def _interrupt_current_operation(self):
        """Cancel any ongoing operation (recording, LLM call, TTS)."""
        # Stop recorder if active
        if self._recorder.is_recording:
            self._recorder.cancel_recording()

        # Stop TTS playback
        self._tts.stop_playback()

        self._set_state(CompanionVoiceState.IDLE)
        logger.info("Interrupt complete, returned to IDLE")

    def _on_audio_level(self, level: float):
        """Audio level callback from recorder (called from recorder thread)."""
        if self._overlay:
            self._overlay.set_audio_level(level)

    # ── VAD (Auto-listen) callbacks ─────────────────────────────────────

    def _on_vad_voice_start(self):
        """VAD detected voice — show cursor and start recording.

        If a TTS response is playing (RESPONDING), interrupt it first
        so user speech takes over immediately.
        """
        logger.info(f"VAD: voice start detected in state {self._state.value}")

        # Always stop any ongoing TTS / LLM work before recording
        if self._state in (CompanionVoiceState.RESPONDING, CompanionVoiceState.PROCESSING):
            self._interrupt_current_operation()

        if self._state == CompanionVoiceState.IDLE:
            self._set_state(CompanionVoiceState.LISTENING)
            if self._overlay:
                self._overlay.show_cursor()
                self._overlay.set_audio_level(0.0)

            # Start recording audio (same as hotkey push-to-talk)
            self._audio_chunks.clear()
            self._transcription_provider = get_available_provider()
            if not self._transcription_provider:
                logger.warning("No transcription provider available!")
                self._set_state(CompanionVoiceState.IDLE)
                return

            self._recorder.start_recording(
                on_audio_level=self._on_audio_level
            )

            # Start transcription session in async loop
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._start_transcription(), self._loop
                )

    def _on_vad_voice_end(self, pcm_data: bytes):
        """VAD detected end of utterance — process audio."""
        logger.info(f"VAD: voice end, captured {len(pcm_data)} bytes")
        if not pcm_data:
            self._set_state(CompanionVoiceState.IDLE)
            return

        self._set_state(CompanionVoiceState.PROCESSING)
        if self._overlay:
            self._overlay.set_audio_level(0.0)

        # Process in async loop
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._process_interaction(pcm_data), self._loop
            )

    # ── Async Transcription ─────────────────────────────────────────────

    async def _start_transcription(self):
        """Start a streaming transcription session."""
        try:
            self._transcription_session = await self._transcription_provider.start_streaming_session(
                on_partial_transcript=self._on_partial_transcript,
            )
            if hasattr(self._transcription_session, 'connect'):
                await self._transcription_session.connect()

            # Feed any already-buffered audio chunks
            for chunk in self._audio_chunks:
                await self._transcription_session.append_audio_buffer(chunk)

        except Exception as e:
            logger.error(f"Failed to start transcription: {e}")
            self._set_state(CompanionVoiceState.IDLE)

    def _on_partial_transcript(self, text: str):
        """Called with partial transcription results."""
        if text:
            logger.debug(f"Partial transcript: {text}")

    # ── Main Interaction Flow ───────────────────────────────────────────

    async def _process_interaction(self, pcm_data: bytes):
        """Full interaction pipeline: transcribe -> capture -> query Claude -> respond."""
        try:
            # 1. Finalize transcription
            transcript = ""
            if self._transcription_session:
                # Send remaining audio
                wav_data = pcm16_to_wav(pcm_data)
                await self._transcription_session.append_audio_buffer(wav_data)

                # Get final transcript
                transcript = await self._transcription_session.request_final_transcript()
                self._transcription_session = None

            if not transcript or not transcript.strip():
                logger.warning("Empty transcript, ignoring")
                self._set_state(CompanionVoiceState.IDLE)
                return

            logger.info(f"Transcript: {transcript}")

            # 2. Capture screens
            images = None
            try:
                images = self._screen.capture_all_screens_base64()
                if images:
                    logger.info(f"Captured {len(images)} screen(s)")
                    # Update scale factors for each captured screen so fly_to_point
                    # can convert image-space coords from LLM to display-space
                    for screen_idx, img in enumerate(images):
                        self._overlay.set_scale_factors(
                            screen_idx,
                            display_width=img.get("display_width", 0),
                            display_height=img.get("display_height", 0),
                            screenshot_width=img.get("screenshot_width", 0),
                            screenshot_height=img.get("screenshot_height", 0),
                        )
                else:
                    logger.warning("No images captured - screen capture returned empty list")
            except Exception as e:
                logger.warning(f"Screen capture failed: {e}")

            # 3. Query Claude with streaming
            full_response = ""
            streaming_text = ""

            def on_text_chunk(chunk: str):
                nonlocal streaming_text, full_response
                full_response += chunk
                streaming_text += chunk
                # Update overlay with streaming text (truncated for display)
                if self._overlay:
                    display = streaming_text
                    if len(display) > 200:
                        display = display[-200:]
                    self._overlay.set_response_text(display)

            response = await self._vision.analyze_with_streaming(
                user_prompt=transcript,
                images=images,
                on_text_chunk=on_text_chunk,
            )

            if not response or not response.strip():
                logger.warning("Empty response from Claude")
                self._set_state(CompanionVoiceState.IDLE)
                return

            # Debug: log full raw response
            logger.info(f"RAW LLM RESPONSE: {response}")

            # 4. Parse scroll commands and execute immediately (before TTS)
            scroll_commands = parse_scroll_commands(response)
            for scroll in scroll_commands:
                _execute_scroll(scroll["direction"], scroll["count"])

            # 5. Parse pointing coordinates
            points = parse_pointing_coordinates(response)
            speak_text = strip_pointing_tags(response)

            # Log response to terminal
            print(f"\n[clicky] {speak_text}\n")

            # 6. Animate cursor to any pointed elements
            logger.info(f"DEBUG: points={points}, _overlay={self._overlay}")
            if points and self._overlay:
                for point in points:
                    logger.info(f"DEBUG: calling fly_to_point x={point['x']} y={point['y']} label={point.get('label','')} screen={point.get('screen',1)}")
                    self._overlay.fly_to_point(
                        x=point["x"],
                        y=point["y"],
                        label=point.get("label", ""),
                        screen=point.get("screen", 1),
                    )

            # 7. Speak the response
            # Release cursor lock so visual cursor tracks real mouse during TTS
            if self._overlay:
                self._overlay.release_cursor_lock()
            self._set_state(CompanionVoiceState.RESPONDING)
            if speak_text.strip():
                await self._tts.speak_text(speak_text)
            else:
                # No text to speak, go back to idle
                self._set_state(CompanionVoiceState.IDLE)

        except Exception as e:
            logger.error(f"Interaction error: {e}", exc_info=True)
            self._set_state(CompanionVoiceState.IDLE)

    def _on_tts_finished(self):
        """Called when TTS playback completes."""
        if self._state == CompanionVoiceState.RESPONDING:
            self._set_state(CompanionVoiceState.IDLE)

    # ── Public API ──────────────────────────────────────────────────────

    def clear_conversation(self):
        """Clear Claude conversation history."""
        self._vision.clear_history()
        logger.info("Conversation history cleared")

    def set_model(self, model: str):
        """Change the Claude model."""
        self._vision._model = model
        logger.info(f"Model changed to: {model}")

    def toggle_mute(self):
        """Toggle TTS mute (placeholder for future)."""
        pass