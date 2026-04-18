"""Multi-monitor screen capture using python-mss and Pillow.

Captures all connected displays as JPEG, labeling each with cursor presence.
Mirrors CompanionScreenCaptureUtility.swift from the original macOS app.
"""

import io
import base64
from dataclasses import dataclass, field

import mss
import numpy as np
from PIL import Image
from pynput import mouse
from typing import List


@dataclass
class ScreenCapture:
    """A single display's captured image."""

    image_data: bytes  # JPEG bytes
    label: str  # e.g. "Display 1" or "Display 1 (with cursor)"
    is_cursor_screen: bool = False
    display_width: int = 0
    display_height: int = 0
    screenshot_width: int = 0
    screenshot_height: int = 0


class ScreenCaptureUtility:
    """Captures all displays, resizes to max dimension, returns JPEG data."""

    def __init__(self, max_dimension: int = 1280, jpeg_quality: int = 80):
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self._mouse_x = 0
        self._mouse_y = 0

        # Track mouse position
        self._mouse_listener = mouse.Listener(
            move=self._on_mouse_move,
        )
        self._mouse_listener.start()

    def _on_mouse_move(self, x: int, y: int):
        self._mouse_x = x
        self._mouse_y = y

    def _get_cursor_screen_index(self, monitors: List[dict]) -> int:
        """Determine which monitor the cursor is on."""
        mx, my = self._mouse_x, self._mouse_y
        for i, mon in enumerate(monitors):
            left = mon.get("left", 0)
            top = mon.get("top", 0)
            right = left + mon.get("width", 0)
            bottom = top + mon.get("height", 0)
            if left <= mx < right and top <= my < bottom:
                return i
        return 0

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image so longest side is max_dimension, maintaining aspect ratio."""
        w, h = img.size
        if max(w, h) <= self.max_dimension:
            return img

        if w >= h:
            new_w = self.max_dimension
            new_h = int(h * (self.max_dimension / w))
        else:
            new_h = self.max_dimension
            new_w = int(w * (self.max_dimension / h))

        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def capture_all_screens(self) -> List[ScreenCapture]:
        """Capture all connected displays as JPEG.

        Returns list of ScreenCapture, sorted with cursor screen first.
        """
        captures = []

        with mss.mss() as sct:
            monitors = sct.monitors  # [0] is all combined, [1:] are individual
            if len(monitors) <= 1:
                # Only the virtual combined monitor, capture it
                monitors_to_capture = [monitors[0]] if monitors else []
            else:
                monitors_to_capture = monitors[1:]  # Skip the combined virtual monitor

            cursor_screen = self._get_cursor_screen_index(monitors_to_capture)

            for i, monitor in enumerate(monitors_to_capture):
                # Capture screenshot
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

                display_width = monitor.get("width", img.width)
                display_height = monitor.get("height", img.height)

                # Resize
                img = self._resize_image(img)

                # Convert to JPEG
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=self.jpeg_quality)
                jpeg_data = buffer.getvalue()

                is_cursor = (i == cursor_screen)
                label = f"Display {i + 1}"
                if is_cursor:
                    label += " (with cursor)"

                captures.append(ScreenCapture(
                    image_data=jpeg_data,
                    label=label,
                    is_cursor_screen=is_cursor,
                    display_width=display_width,
                    display_height=display_height,
                    screenshot_width=img.width,
                    screenshot_height=img.height,
                ))

        # Sort: cursor screen first
        captures.sort(key=lambda c: (0 if c.is_cursor_screen else 1))

        return captures

    def capture_all_screens_base64(self) -> List[dict]:
        """Capture all screens and return as base64-encoded dicts for API calls.

        Returns list of dicts with 'data', 'media_type', 'label', 'is_cursor_screen'.
        """
        captures = self.capture_all_screens()
        result = []
        for capture in captures:
            b64 = base64.standard_b64encode(capture.image_data).decode("utf-8")
            result.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
                "label": capture.label,
                "is_cursor_screen": capture.is_cursor_screen,
                "display_width": capture.display_width,
                "display_height": capture.display_height,
                "screenshot_width": capture.screenshot_width,
                "screenshot_height": capture.screenshot_height,
            })
        return result

    def cleanup(self):
        """Stop mouse listener."""
        if self._mouse_listener:
            self._mouse_listener.stop()