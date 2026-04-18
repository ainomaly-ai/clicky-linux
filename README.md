# Clicky Linux

An AI buddy that lives next to your cursor. It can see your screen, talk to you, and point at stuff. Linux port of the original [Clicky](https://github.com/farzaa/clicky) by Farzaa.

## Features

- Push-to-talk voice input (Ctrl+Alt hold-to-talk, or hands-free auto-listen VAD mode)
- Screen capture + vision analysis (local or cloud LLMs)
- Streaming text responses with **free neural TTS** (Edge TTS — no API key needed)
- Blue cursor overlay that flies to UI elements the AI points at
- System tray icon with state-aware control panel
- Multiple LLM backends: local (Ollama, llama.cpp) or cloud (Anthropic, OpenAI)
- Multiple TTS backends: Edge TTS (free), ElevenLabs (paid), pyttsx3 (offline/robotic)
- Multiple transcription backends: AssemblyAI (cloud) or faster-whisper (local)
- Hermes agent integration: use Hermes as a guiding agent with tools, memory, and skills
- Local API proxy support (keys never leave the proxy)

## Completely Free (No API Keys Required)

Clicky Linux runs **entirely offline** with free local models:

| Component | Free Option | Notes |
|-----------|-------------|-------|
| LLM | Ollama or llama.cpp | Any GGUF model (Qwen, Gemma, Llama, etc.) |
| TTS | **Edge TTS** | Neural voices, no API key, great quality |
| Transcription | faster-whisper | Local Whisper, no API key |
| Vision | Ollama (LLaVA) | Or use cloud if preferred |

**Zero API costs. Your data never leaves your machine.**

### Recommended Free Setup

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:27b
TTS_BACKEND=edge
EDGE_TTS_VOICE=en-GB-SoniaNeural
AUTO_LISTEN=true
```

## Setup

### Prerequisites

- Python 3.10+
- Linux with X11 or Wayland
- Audio: PulseAudio or PipeWire
- For cloud features (optional): Anthropic and/or OpenAI API key

### Install

```bash
cd clicky-linux
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Edit .env with your API keys (or use the free local setup above)
```

### Run

```bash
clicky
```

Or with the proxy server separately (for shared API keys across machines):

```bash
clicky-proxy    # Start the API proxy on port 8787
clicky          # Start the app (connects to proxy)
```

## Hotkey

Default: **Ctrl+Alt** — hold to talk, release to send.

Alternatively, enable hands-free auto-listen mode (`AUTO_LISTEN=true`) which uses voice activity detection — no hotkey needed.

## Architecture

```
Push-to-talk (Ctrl+Alt) or Auto-listen (VAD)
  -> Hotkey monitor detects press/release
  -> Audio recorder captures mic input (sounddevice)
  -> Audio -> AssemblyAI WebSocket or faster-whisper (local)
  -> Final transcript + screenshots -> LLM (local Ollama/llama.cpp or cloud)
  -> LLM responds with text + optional [POINT:x,y:label] tags
  -> TTS speaks response (Edge TTS, ElevenLabs, or pyttsx3 fallback)
  -> Cursor overlay flies to pointed coordinates (PyQt6)
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | Backend: `ollama`, `llamacpp`, `anthropic`, `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:27b` | Ollama model name |
| `LLAMACPP_BASE_URL` | `http://localhost:8899` | llama.cpp server URL |
| `LLAMACPP_MODEL` | _(empty)_ | llama.cpp model file |
| `LLAMACPP_DISABLE_THINKING` | `true` | Strip thinking tokens from llama.cpp output |
| `TTS_BACKEND` | `edge` | Backend: `edge`, `elevenlabs`, `pyttsx3` |
| `EDGE_TTS_VOICE` | `en-GB-SoniaNeural` | Edge TTS voice (see Edge TTS voice list) |
| `ANTHROPIC_API_KEY` | _(empty)_ | Required only for cloud Anthropic |
| `ASSEMBLYAI_API_KEY` | _(empty)_ | Required only for AssemblyAI transcription |
| `ELEVENLABS_API_KEY` | _(empty)_ | Required only for ElevenLabs TTS |
| `ELEVENLABS_VOICE_ID` | _(empty)_ | ElevenLabs voice ID |
| `OPENAI_API_KEY` | _(empty)_ | Required for OpenAI backend/vision |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `USE_PROXY` | `false` | Route API calls through local proxy |
| `PROXY_URL` | `http://localhost:8787` | Proxy server URL |
| `HOTKEY_COMBO` | `ctrl+alt` | Push-to-talk hotkey |
| `AUTO_LISTEN` | `false` | Enable voice activity detection (no hotkey) |
| `AUTO_LISTEN_THRESHOLD` | `0.05` | VAD sensitivity (lower = more sensitive) |
| `AUTO_LISTEN_TIMEOUT` | `3.0` | Seconds of silence before processing |
| `CLAUDE_MODEL` | _(used only with proxy)_ | Claude model via proxy |
| `WHISPER_MODEL` | `tiny` | Local whisper model for fallback STT |
| `GUIDING_AGENT` | `llm` | `llm` (direct) or `hermes` (agent with tools) |
| `HERMES_AGENT_SKILLS` | _(see .env)_ | Comma-separated skills for Hermes mode |

## Linux Permissions

Clicky needs access to:

- **Microphone** — PulseAudio/PipeWire recording
- **Screen recording** — X11/Wayland screen capture (may need `xdg-desktop-portal` on Wayland)
- **Input monitoring** — Global hotkey (udev/input group on some distros)

## License

MIT License — see [LICENSE](LICENSE). Free to use, modify, and distribute.

## Credits

- **Original Clicky**: [Farzaa](https://github.com/farzaa/clicky) — the macOS/Windows version this project is ported from
- **Edge TTS**: Microsoft Azure Cognitive Services (free neural voices)
- **faster-whisper**: [Guillaumedteisseire/faster-whisper](https://github.com/Guillaumedteisseire/faster-whisper) — efficient local transcription
- **PyQt6**: Riverbank Computing — Qt bindings for Python
- **llama.cpp**: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) — pure C/C++ LLM inference
- **Ollama**: [ollama/ollama](https://github.com/ollama/ollama) — local LLM server
