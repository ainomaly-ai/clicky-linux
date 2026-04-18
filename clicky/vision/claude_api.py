"""Claude API client with SSE streaming.

Sends transcript + screenshots to Claude and streams text responses.
Mirrors ClaudeAPI.swift from the original macOS app.
"""

import json
import logging
import re
from typing import Callable, AsyncIterator, Optional, List

import httpx

from clicky.config import config

logger = logging.getLogger(__name__)

# System prompt for Clicky (matches original macOS app)
SYSTEM_PROMPT = """you're clicky. a friendly, casual ai buddy that lives next to the user's cursor on their linux desktop. you can see their screen and help them with whatever they're working on.

key behaviors:
- be casual and lowercase. like talking to a friend, not a robot
- keep responses short and conversational - this gets spoken aloud via tts
- don't use markdown, formatting, or special characters - just plain text
- if you can see something relevant on their screen, reference it naturally
- you can point at UI elements by embedding [POINT:x,y:label:screenN] tags in your response
  - x and y are **normalized coordinates from 0.0 to 1.0**, where (0,0) is the top-left of the ENTIRE screen/screenshot and (1,1) is the bottom-right
  - IMPORTANT: (0,0) is the top-left corner of the full screenshot, which shows the entire desktop. Do NOT use coordinates relative to a window or browser content area — use the full screenshot bounds
  - label is a short description of what you're pointing at
  - screenN is the display number (1-based)
  - example: "click the save button here [POINT:0.35,0.22:Save button:1]"
- only use [POINT] tags when actually pointing at something visible in the screenshot
- be helpful, direct, and friendly
- if they ask about code, explain clearly but briefly
- if something's wrong on screen, point it out naturally
- you can scroll the page by embedding [SCROLL:down] or [SCROLL:up] tags
  - [SCROLL:down] scrolls down one page
  - [SCROLL:up:3] scrolls up 3 pages
  - you can also use [SCROLL:0.5] for half a page, [SCROLL:0.3] for 30% scroll
  - only use scroll tags when the user asks to scroll"""


class ClaudeAPI:
    """Claude vision API client with SSE streaming."""

    def __init__(self):
        self._api_key = config.ANTHROPIC_API_KEY
        self._model = config.CLAUDE_MODEL
        self._max_tokens = 1024
        self._conversation_history: List[dict] = []

    @property
    def api_url(self) -> str:
        return config.chat_url

    def _get_headers(self) -> dict:
        headers = {
            "content-type": "application/json",
        }
        if config.USE_PROXY:
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["x-api-key"] = self._api_key
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()

    async def analyze_with_streaming(
        self,
        user_prompt: str,
        images: Optional[List[dict]] = None,
        on_text_chunk: Callable[[str], None] = None,
    ) -> str:
        """Send a message to Claude with optional images and stream the response.

        Args:
            user_prompt: The user's transcript/message
            images: List of image dicts from ScreenCaptureUtility.capture_all_screens_base64()
            on_text_chunk: Callback for each text chunk as it arrives

        Returns:
            The complete response text
        """
        # Build message content
        content = []

        # Add images first
        if images:
            for img in images:
                content.append({
                    "type": "image",
                    "source": img["source"],
                })

        # Add text prompt
        content.append({
            "type": "text",
            "text": user_prompt,
        })

        # Build conversation messages
        messages = list(self._conversation_history)
        messages.append({"role": "user", "content": content})

        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "stream": True,
        }

        full_response = ""

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream(
                "POST",
                self.api_url,
                json=body,
                headers=self._get_headers(),
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"Claude API error {response.status_code}: {error_body.decode()}")
                    raise Exception(f"Claude API error: {response.status_code}")

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type", "")

                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                full_response += text
                                if on_text_chunk:
                                    on_text_chunk(text)

                        elif event_type == "message_stop":
                            break

                        elif event_type == "error":
                            error = data.get("error", {})
                            logger.error(f"Claude stream error: {error}")
                            raise Exception(f"Claude error: {error.get('message', 'unknown')}")

        # Save to conversation history
        self._conversation_history.append({"role": "user", "content": content})
        self._conversation_history.append({"role": "assistant", "content": full_response})

        # Keep only last 10 exchanges (20 messages)
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        return full_response

    async def analyze_without_streaming(
        self,
        user_prompt: str,
        images: Optional[List[dict]] = None,
    ) -> str:
        """Non-streaming version for element detection etc."""
        content = []

        if images:
            for img in images:
                content.append({
                    "type": "image",
                    "source": img["source"],
                })

        content.append({"type": "text", "text": user_prompt})

        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            response = await client.post(
                self.api_url,
                json=body,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                raise Exception(f"Claude API error: {response.status_code} {response.text}")

            data = response.json()
            return data.get("content", [{}])[0].get("text", "")


def parse_pointing_coordinates(response_text: str) -> List[dict]:
    """Parse [POINT:x,y:label:screenN] tags from Claude's response.

    Supports both pixel coordinates (e.g., POINT:450,320) and
    normalized 0-1 coordinates (e.g., POINT:0.35,0.22).
    """
    points = []
    # Support both integer pixels (e.g. 450) and normalized floats (e.g. 0.35)
    # Label group matches everything up to "]" (non-greedy) so colons in the label
    # (e.g. URLs like https://...) don't break field parsing.
    # No $ anchor — POINT tags can appear anywhere in the response (mid-sentence
    # or end-of-sentence), e.g. "click it [POINT:0.08,0.28:Save:1] to continue".
    pattern = r'\[POINT:([\d.]+),([\d.]+):(.*?)(?::(\d+))?\]'
    for match in re.finditer(pattern, response_text, re.MULTILINE | re.IGNORECASE):
        points.append({
            "x": float(match.group(1)),  # float to support both pixel and normalized
            "y": float(match.group(2)),
            "label": match.group(3) or "",
            "screen": int(match.group(4)) if match.group(4) else 1,
        })
    return points


def strip_pointing_tags(text: str) -> str:
    """Remove [POINT:...] and [SCROLL:...] tags and clean markdown for TTS."""
    # Use non-anchored pattern — POINT tags can appear anywhere in the sentence
    text = re.sub(r'\[POINT:[\d.]+,[\d.]+:(.*?)(?::\d+)?\]', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\[SCROLL:[\w.]+(?::[\d.]+)?\]', '', text)

    # Strip markdown formatting for natural TTS reading
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold** -> text
    text = re.sub(r'\*(.+?)\*', r'\1', text)        # *italic* -> text
    text = re.sub(r'__(.+?)__', r'\1', text)        # __bold__ -> text
    text = re.sub(r'_(.+?)_', r'\1', text)          # _italic_ -> text
    text = re.sub(r'~~(.+?)~~', r'\1', text)        # ~~strikethrough~~ -> text
    text = re.sub(r'`(.+?)`', r'\1', text)          # `code` -> text
    text = re.sub(r'^#{1,6}\s+', '', text)          # # headings -> text
    text = re.sub(r'^\s*[-*+]\s+', '', text)        # - bullet points -> text
    text = re.sub(r'^\s*\d+\.\s+', '', text)        # 1. numbered lists -> text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text) # [text](url) -> text

    return text.strip()


def parse_scroll_commands(response_text: str) -> List[dict]:
    """Parse [SCROLL:direction:count] tags from LLM response.

    Examples:
        [SCROLL:down]    -> scroll down 1x (one page)
        [SCROLL:up:3]    -> scroll up 3x
        [SCROLL:down:2]  -> scroll down 2x
        [SCROLL:0.5]     -> scroll down 50% of page
        [SCROLL:up:0.3]  -> scroll up 30% of page
    """
    commands = []
    pattern = r'\[SCROLL:([\w.]+)(?::([\d.]+))?\]'
    for match in re.finditer(pattern, response_text, re.IGNORECASE):
        direction = match.group(1).lower()
        count_str = match.group(2)

        if count_str and '.' in count_str:
            # Fractional scroll (e.g., 0.5 = 50% of page)
            commands.append({"direction": direction, "count": float(count_str)})
        elif direction in ("up", "down"):
            count = int(count_str) if count_str else 1
            commands.append({"direction": direction, "count": count})
        elif direction.replace('.', '', 1).isdigit():
            # Just a number like [SCROLL:0.5] - treat as fractional down scroll
            commands.append({"direction": "down", "count": float(direction)})
    return commands