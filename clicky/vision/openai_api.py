"""OpenAI-compatible vision API client.

Supports any OpenAI-compatible endpoint (vLLM, llama.cpp, custom servers)
via OPENAI_BASE_URL + OPENAI_API_KEY configuration.
"""

import json
import logging
import os
from typing import Callable, Optional, List

import httpx

from clicky.config import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """you're clicky. a friendly, casual ai buddy that lives next to the user's cursor on their linux desktop. you can see their screen and help them with whatever they're working on.

key behaviors:
- be casual and lowercase. like talking to a friend, not a robot
- keep responses short and conversational - this gets spoken aloud via tts
- do NOT use any markdown, **bold**, *italic*, # headings, `code`, bullet points, or any formatting - just plain text
- if you can see something relevant on their screen, reference it naturally
- you can point at UI elements by embedding [POINT:x,y:label:screenN] tags in your response
  - x and y are normalized coordinates from 0.0 to 1.0, where (0,0) is the top-left of the ENTIRE screenshot and (1,1) is the bottom-right
  - IMPORTANT: (0,0) is the top-left corner of the full screenshot. Do NOT use coordinates relative to a window or browser content area
  - label is a short description of what you're pointing at
  - screenN is the display number (1-based)
  - example: "click the save button here [POINT:0.35,0.22:Save button:1]"
- only use [POINT] tags when actually pointing at something visible in the screenshot
- also use [SCROLL:up:count] or [SCROLL:down:count] to scroll when needed
- be helpful, direct, and friendly
- if they ask about code, explain clearly but briefly"""


def _convert_images_to_openai(images: List[dict]) -> List[dict]:
    """Convert Clicky image format (Anthropic-style) to OpenAI format.

    Input:  [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}]
    Output: [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
    """
    result = []
    for img in images:
        source = img.get("source", {})
        if source.get("type") == "base64":
            media_type = source.get("media_type", "image/jpeg")
            b64_data = source.get("data", "")
            result.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{b64_data}",
                    "detail": "high",
                },
            })
    return result


class OpenAIVisionAPI:
    """OpenAI-compatible vision API client.

    Works with any server that exposes /v1/chat/completions with SSE streaming.
    Configure via OPENAI_BASE_URL and OPENAI_API_KEY in .env.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self._base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        # Normalize: strip trailing /v1 since we always append it
        if self._base_url.endswith("/v1"):
            self._base_url = self._base_url[:-3]
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._api_key = api_key or config.OPENAI_API_KEY
        self._conversation_history: List[dict] = []

    @property
    def api_url(self) -> str:
        return f"{self._base_url}/v1/chat/completions"

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()

    async def analyze_with_streaming(
        self,
        user_prompt: str,
        images: Optional[List[dict]] = None,
        on_text_chunk: Callable[[str], None] = None,
    ) -> str:
        """Send a message with optional images and stream the response.

        Args:
            user_prompt: The user's transcript/message
            images: List of image dicts from ScreenCaptureUtility.capture_all_screens_base64()
            on_text_chunk: Callback for each text chunk as it arrives

        Returns:
            The complete response text
        """
        if not self._api_key:
            logger.warning("OPENAI_API_KEY not configured, using placeholder")
            self._api_key = "placeholder"

        content = []
        if images:
            content.extend(_convert_images_to_openai(images))
        content.append({"type": "text", "text": user_prompt})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": content})

        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 1024,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        full_response = ""
        streaming_text = ""

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream("POST", self.api_url, json=body, headers=headers) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise Exception(f"OpenAI API error: {response.status_code} {text.decode()}")

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # OpenAI SSE format: choices[0].delta.content
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text_chunk = delta.get("content", "")
                    if text_chunk:
                        streaming_text += text_chunk
                        full_response += text_chunk
                        if on_text_chunk:
                            on_text_chunk(text_chunk)

        # Update conversation history (keep last 20 messages)
        self._conversation_history.append({"role": "user", "content": user_prompt})
        self._conversation_history.append({"role": "assistant", "content": full_response})
        if len(self._conversation_history) > 40:
            self._conversation_history = self._conversation_history[-40:]

        return full_response

    async def analyze_without_streaming(
        self,
        user_prompt: str,
        images: Optional[List[dict]] = None,
    ) -> str:
        """Non-streaming version (calls analyze_with_streaming with no callback)."""
        return await self.analyze_with_streaming(user_prompt, images, on_text_chunk=None)