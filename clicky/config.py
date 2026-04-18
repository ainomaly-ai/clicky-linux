"""Configuration management for Clicky.

Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load .env from project directory
_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


class Config:
    """Application configuration. All settings from environment variables."""

    # --- LLM Backend ---
    # Options: "ollama", "llamacpp", "anthropic", "openai"
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")

    # --- Ollama ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:27b")

    # --- llama.cpp server ---
    LLAMACPP_BASE_URL: str = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8899")
    LLAMACPP_MODEL: str = os.getenv("LLAMACPP_MODEL", "")
    LLAMACPP_DISABLE_THINKING: bool = os.getenv("LLAMACPP_DISABLE_THINKING", "true").lower() in ("true", "1", "yes")

    # --- TTS Backend ---
    # Options: "elevenlabs" (API key needed), "pyttsx3" (offline, robotic), "edge" (free neural, best quality)
    TTS_BACKEND: str = os.getenv("TTS_BACKEND", "edge")

    # --- Edge TTS (free neural voices, no API key) ---
    EDGE_TTS_VOICE: str = os.getenv("EDGE_TTS_VOICE", "en-GB-SoniaNeural")

    # --- API Keys ---
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ASSEMBLYAI_API_KEY: str = os.getenv("ASSEMBLYAI_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # --- Proxy ---
    USE_PROXY: bool = os.getenv("USE_PROXY", "false").lower() in ("true", "1", "yes")
    PROXY_URL: str = os.getenv("PROXY_URL", "http://localhost:8787")
    PROXY_PORT: int = int(os.getenv("PROXY_PORT", "8787"))

    # --- Hotkey ---
    HOTKEY_COMBO: str = os.getenv("HOTKEY_COMBO", "ctrl+alt")

    # --- Model ---
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

    # --- Transcription ---
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")

    # --- Audio ---
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1

    # --- Auto-listen (VAD mode — no hotkey needed) ---
    AUTO_LISTEN: bool = os.getenv("AUTO_LISTEN", "false").lower() in ("true", "1", "yes")
    AUTO_LISTEN_THRESHOLD: float = float(os.getenv("AUTO_LISTEN_THRESHOLD", "0.05"))  # lower = more sensitive
    AUTO_LISTEN_TIMEOUT: float = float(os.getenv("AUTO_LISTEN_TIMEOUT", "3.0"))  # seconds of silence before processing

    # --- Screen Capture ---
    SCREENSHOT_MAX_DIMENSION: int = 1280
    SCREENSHOT_JPEG_QUALITY: int = 80

    # --- Guiding Agent ---
    # Options: "llm" (direct LLM backend), "hermes" (Hermes agent with tools/memory/skills)
    GUIDING_AGENT: str = os.getenv("GUIDING_AGENT", "llm")

    # Skills loaded for Hermes agent (comma-separated, only used when GUIDING_AGENT=hermes)
    HERMES_AGENT_SKILLS: str = os.getenv(
        "HERMES_AGENT_SKILLS",
        "writing-plans,code-review,project-onboard,graphify",
    )

    # --- Derived ---
    @property
    def chat_url(self) -> str:
        if self.USE_PROXY:
            return f"{self.PROXY_URL}/chat"
        return "https://api.anthropic.com/v1/messages"

    @property
    def tts_url(self) -> str:
        if self.USE_PROXY:
            return f"{self.PROXY_URL}/tts"
        return f"https://api.elevenlabs.io/v1/text-to-speech/{self.ELEVENLABS_VOICE_ID}"

    @property
    def transcribe_token_url(self) -> str:
        if self.USE_PROXY:
            return f"{self.PROXY_URL}/transcribe-token"
        return "https://streaming.assemblyai.com/v3/token"

    def validate(self) -> List[str]:
        """Return list of missing required configuration."""
        missing = []
        backend = self.LLM_BACKEND.lower()
        if backend == "anthropic" and not self.ANTHROPIC_API_KEY and not self.USE_PROXY:
            missing.append("ANTHROPIC_API_KEY")
        elif backend == "openai" and not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        elif backend in ("ollama", "llamacpp"):
            # No API key needed for local backends
            pass
        if self.GUIDING_AGENT not in ("llm", "hermes"):
            missing.append(f"GUIDING_AGENT must be 'llm' or 'hermes', got '{self.GUIDING_AGENT}'")
        return missing


config = Config()
