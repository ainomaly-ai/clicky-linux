"""Ollama vision API client using OpenAI-compatible endpoint.

Works with any Ollama model that supports vision (images).
Uses /v1/chat/completions for standard OpenAI-compatible streaming.
"""

import json
import logging
from typing import Callable, Optional, List

import httpx

from clicky.config import config
from clicky.vision.claude_api import SYSTEM_PROMPT, parse_pointing_coordinates, strip_pointing_tags

logger = logging.getLogger(__name__)


def _convert_images_to_openai(images: List[dict]) -> List[dict]:
    """Convert Clicky image format (Anthropic-style) to OpenAI/Ollama format.

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


class OllamaAPI:
    """OpenAI-compatible vision API client.

    Works with Ollama, llama.cpp server, and any backend that
    exposes /v1/chat/completions with SSE streaming.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, extra_body: Optional[dict] = None):
        self._base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._model = model or config.OLLAMA_MODEL
        self._max_tokens = 1024
        self._extra_body = extra_body or {}  # Merged into request body (e.g. chat_template_kwargs)
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
        """Send a message to Ollama with optional images and stream the response.

        Same interface as ClaudeAPI.analyze_with_streaming().

        Args:
            user_prompt: The user's transcript/message
            images: List of image dicts from ScreenCaptureUtility.capture_all_screens_base64()
            on_text_chunk: Callback for each text chunk as it arrives

        Returns:
            The complete response text
        """
        # Build message content
        content = []

        # Add images in OpenAI format
        if images:
            openai_images = _convert_images_to_openai(images)
            content.extend(openai_images)

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
            "messages": messages,
            "stream": True,
        }

        # If no model specified (e.g. single-model llama.cpp server), omit model field
        if not self._model:
            del body["model"]

        # Merge extra body params (e.g. chat_template_kwargs for llama.cpp thinking models)
        body.update(self._extra_body)

        # Add system prompt as a system message (Ollama/OpenAI style)
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        full_response = ""

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream(
                "POST",
                self.api_url,
                json=body,
                headers={"content-type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"Ollama API error {response.status_code}: {error_body.decode()}")
                    raise Exception(f"Ollama API error: {response.status_code} - {error_body.decode()[:200]}")

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # OpenAI SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # OpenAI streaming format: choices[0].delta.content
                        # Also handle reasoning_content from thinking models (Qwen3, etc.)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            # Main content
                            text = delta.get("content", "")
                            if text:
                                full_response += text
                                if on_text_chunk:
                                    on_text_chunk(text)
                            # Reasoning/thinking content (skip for TTS/display but log)
                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                logger.debug(f"Reasoning: {reasoning[:100]}")

                        # Check finish_reason
                        if choices and choices[0].get("finish_reason") == "stop":
                            break

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
            openai_images = _convert_images_to_openai(images)
            content.extend(openai_images)

        content.append({"type": "text", "text": user_prompt})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": content})

        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            "stream": False,
        }

        # If no model specified (e.g. single-model llama.cpp server), omit model field
        if not self._model:
            del body["model"]

        # Merge extra body params
        body.update(self._extra_body)

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            response = await client.post(
                self.api_url,
                json=body,
                headers={"content-type": "application/json"},
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} {response.text}")

            data = response.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            # Handle reasoning_content from thinking models
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                logger.debug(f"Reasoning: {reasoning[:200]}")
            return msg.get("content", "")