"""Element location detector using Claude Computer Use API.

Detects UI element coordinates in screenshots so the cursor overlay
can point at them. Mirrors ElementLocationDetector.swift from the original macOS app.
"""

import json
import logging
import base64
from io import BytesIO

from PIL import Image

from clicky.config import config
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Standard resolutions Claude Computer Use supports
STANDARD_RESOLUTIONS = [
    (1024, 768),
    (1280, 800),
    (1366, 768),
]


class ElementLocationDetector:
    """Uses Claude's Computer Use API to detect UI element locations in screenshots."""

    def __init__(self):
        self._api_key = config.ANTHROPIC_API_KEY

    def _find_closest_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Find the closest standard resolution for Computer Use."""
        best = STANDARD_RESOLUTIONS[0]
        best_diff = float("inf")
        for rw, rh in STANDARD_RESOLUTIONS:
            diff = abs(rw - width) + abs(rh - height)
            if diff < best_diff:
                best_diff = diff
                best = (rw, rh)
        return best

    def _resize_to_standard(self, image_data: bytes) -> Tuple[bytes, int, int, float, float]:
        """Resize screenshot to closest standard resolution for Computer Use.

        Returns:
            (resized_jpeg_bytes, target_width, target_height, x_scale, y_scale)
        """
        img = Image.open(BytesIO(image_data))
        orig_w, orig_h = img.size

        target_w, target_h = self._find_closest_resolution(orig_w, orig_h)
        x_scale = orig_w / target_w
        y_scale = orig_h / target_h

        resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        buf = BytesIO()
        resized.save(buf, format="JPEG", quality=80)
        return buf.getvalue(), target_w, target_h, x_scale, y_scale

    async def detect_element(
        self,
        image_data: bytes,
        element_description: str,
    ) -> Optional[dict]:
        """Detect a UI element's coordinates in a screenshot.

        Args:
            image_data: Screenshot JPEG bytes
            element_description: What to look for (e.g. "save button")

        Returns:
            Dict with 'x', 'y' (in original screen coordinates) or None
        """
        import httpx

        resized_data, target_w, target_h, x_scale, y_scale = self._resize_to_standard(image_data)
        b64 = base64.standard_b64encode(resized_data).decode("utf-8")

        # Computer Use tool definition
        tool_def = {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": target_w,
            "display_height_px": target_h,
        }

        body = {
            "model": config.CLAUDE_MODEL,
            "max_tokens": 1024,
            "tools": [tool_def],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Find the {element_description} on screen and click on it.",
                        },
                    ],
                },
            ],
        }

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "computer-use-2025-11-24",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=body,
                headers=headers,
            )

            if response.status_code != 200:
                logger.error(f"Computer Use API error: {response.status_code}")
                return None

            data = response.json()

            # Parse tool_use response for click coordinates
            for block in data.get("content", []):
                if block.get("type") == "tool_use":
                    tool_input = block.get("input", {})
                    if tool_input.get("action") == "left_click":
                        coord = tool_input.get("coordinate", [])
                        if len(coord) == 2:
                            # Scale back to original coordinates
                            orig_x = int(coord[0] * x_scale)
                            orig_y = int(coord[1] * y_scale)
                            return {"x": orig_x, "y": orig_y}

        return None