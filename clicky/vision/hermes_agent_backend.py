"""Hermes agent backend for clicky-linux guiding agent.

Spawns a Hermes agent subprocess that receives transcript + screenshots
and returns a text response with optional [POINT:x,y:label:screen] tags.
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, List

from clicky.config import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You're clicky. A friendly AI buddy living next to the user's cursor on their Linux desktop. You can see their screen and help them with whatever they're working on.

Key behaviors:
- Be casual and lowercase. Like talking to a friend, not a robot.
- Keep responses short and conversational -- this gets spoken aloud via TTS.
- Don't use markdown, formatting, or special characters -- just plain text.
- If you can see something relevant on their screen, reference it naturally.
- You can point at UI elements by embedding [POINT:x,y:label:screenN] tags in your response.
  - x and y are pixel coordinates relative to the display.
  - label is a short description of what you're pointing at.
  - screenN is the display number (1-based).
  - example: "click the save button here [POINT:450,320:Save button:1]"
- Only use [POINT] tags when actually pointing at something visible.
- Be helpful, direct, and friendly.
- You have access to tools (terminal, files, web search). Use them when helpful.

Available tools:
- Terminal: Run commands on the user's desktop
- Files: Read and write files
- Web search: Search the internet for information
- Memory: Remember past interactions

When you need to use a tool, describe what you'd do and why. The user can see your actions."""


class HermesAgentBackend:
    """Hermes agent as a guiding agent backend.

    Spawns a Hermes agent subprocess, sends it the user transcript +
    screenshots, and returns the agent's response. The agent is expected
    to return text with optional [POINT:x,y:label:screenN] coordinate tags.
    """

    def __init__(self, agent_name: str = "clicky"):
        self._agent_name = agent_name
        self._conversation_history: List[dict] = []
        self._max_tokens = 1024
        self._agent_binary = self._find_hermes_binary()
        self._skills = config.HERMES_AGENT_SKILLS if hasattr(config, "HERMES_AGENT_SKILLS") else "writing-plans,code-review,project-onboard,graphify"

    def _find_hermes_binary(self) -> str:
        """Find the hermes agent binary."""
        # Check custom env var first
        custom = os.getenv("HERMES_AGENT_BINARY")
        if custom and os.path.exists(custom):
            return custom

        # Check common locations
        for path in [
            os.path.expanduser("~/.local/bin/hermes"),
            os.path.expanduser("~/hermes-agent/hermes"),
        ]:
            if os.path.exists(path):
                return path

        # Check PATH
        if self._which("hermes"):
            return "hermes"

        raise RuntimeError(
            "Hermes agent binary not found. Install hermes-agent or set HERMES_AGENT_BINARY env var."
        )

    @staticmethod
    def _which(cmd: str) -> bool:
        """Check if a command is in PATH."""
        return subprocess.run(
            ["which", cmd], capture_output=True
        ).returncode == 0

    def _build_prompt(self, user_text: str, images: Optional[List[dict]]) -> str:
        """Build the prompt for the agent, including images."""
        prompt_parts = []

        # Add current user message
        prompt_parts.append(f"User says: {user_text}")

        # Add conversation history (last 5 exchanges)
        if self._conversation_history:
            prompt_parts.append("\n=== PREVIOUS CONVERSATION ===")
            for msg in self._conversation_history[-10:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                prompt_parts.append(f"{role}: {content}")

        # Add images (write to temp files, reference paths)
        if images:
            prompt_parts.append("\n=== SCREENSHOT DATA ===")
            temp_paths = []
            for i, img in enumerate(images):
                source = img.get("source", {})
                if source.get("type") == "base64":
                    data = source.get("data", "")
                    mime = source.get("media_type", "image/jpeg")
                    ext = "jpg" if "jpeg" in mime else "png"
                    fd, img_path = tempfile.mkstemp(
                        prefix=f"clicky_screenshot_{i+1}_", suffix=f".{ext}"
                    )
                    os.close(fd)
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(data))
                    temp_paths.append(img_path)
                    prompt_parts.append(
                        f"Screenshot {i+1} saved to: {img_path} "
                        f"(format: {mime})"
                    )

        prompt_parts.append(
            "\nRespond as clicky. Be brief and conversational. "
            "Use [POINT:x,y:label:screenN] tags when pointing at UI elements."
        )

        return "\n".join(prompt_parts), temp_paths if images else []

    async def analyze_with_streaming(
        self,
        user_prompt: str,
        images: Optional[List[dict]] = None,
        on_text_chunk: Callable[[str], None] = None,
    ) -> str:
        """Send transcript + screenshots to Hermes agent and get response.

        Streams output incrementally by reading stdout line-by-line as the
        subprocess writes it, calling on_text_chunk for each chunk so callers
        (TTS, cursor overlay) can react in real-time.
        """
        prompt, temp_paths = self._build_prompt(user_prompt, images)

        # Build the agent command
        cmd = [
            self._agent_binary,
            "chat",
            "-q",
            prompt,
            "--skills", self._skills,
        ]

        logger.info(f"Running Hermes agent: {' '.join(cmd[:4])}...")

        # Run the agent as a subprocess (with timeout)
        try:
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            logger.error("Hermes agent timed out after 120s")
            raise RuntimeError("Hermes agent timed out after 120 seconds")

        # Read stdout incrementally so on_text_chunk fires as tokens arrive
        full_response = ""
        accumulated = ""
        buf_size = 64

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(buf_size),
                        timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    logger.error("Hermes agent timed out while reading output")
                    process.kill()
                    raise RuntimeError("Hermes agent timed out after 120 seconds")

                if not chunk:
                    break

                text = chunk.decode()
                accumulated += text

                # Emit the raw chunk so TTS/overlay can react immediately
                if on_text_chunk:
                    on_text_chunk(text)

                full_response += text

                # Also print to stdout so user sees streaming tokens in terminal
                print(text, end="", flush=True)

        except Exception as e:
            logger.error(f"Error reading Hermes output: {e}")
            process.kill()
            raise

        # Drain stderr for error reporting
        stderr = await process.stderr.read()
        if process.returncode != 0:
            error_msg = stderr.decode()[:500]
            logger.error(f"Hermes agent error (exit {process.returncode}): {error_msg}")
            raise RuntimeError(f"Hermes agent failed (exit {process.returncode}): {error_msg}")

        response = full_response.strip()

        # Save to conversation history
        self._conversation_history.append({"role": "user", "content": user_prompt})
        self._conversation_history.append({"role": "assistant", "content": response})

        # Keep only last 10 exchanges
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        # Clean up temp files
        for path in temp_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        return response

    async def analyze_without_streaming(
        self,
        user_prompt: str,
        images: Optional[List[dict]] = None,
    ) -> str:
        """Non-streaming version (same as streaming for agent)."""
        return await self.analyze_with_streaming(user_prompt, images)

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()
