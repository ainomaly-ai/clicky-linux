"""Global hotkey monitor using pynput.

Detects push-to-talk shortcut (default: Ctrl+Alt) system-wide.
Mirrors GlobalPushToTalkShortcutMonitor.swift from the original macOS app.
"""

import logging
import threading
from enum import Enum
from typing import Callable, Optional, Set

from pynput import keyboard

logger = logging.getLogger(__name__)


class ShortcutTransition(Enum):
    PRESSED = "pressed"
    RELEASED = "released"


class GlobalHotkeyMonitor:
    """Monitors a modifier-key combination system-wide for push-to-talk.

    Detects when all specified modifier keys are held down simultaneously,
    and when any of them is released.
    """

    def __init__(
        self,
        combo: str = "ctrl+alt",
        on_transition: Callable[[ShortcutTransition], None] = None,
    ):
        """Initialize the hotkey monitor.

        Args:
            combo: Modifier key combination, e.g. "ctrl+alt", "ctrl+shift"
            on_transition: Callback for press/release transitions
        """
        self.combo = combo
        self.on_transition = on_transition

        # Parse combo string into pynput key set
        self._target_keys = self._parse_combo(combo)
        self._currently_pressed: Set[keyboard.Key] = set()
        self._was_active = False
        self._listener: Optional[keyboard.Listener] = None
        self._lock = threading.Lock()

    def _parse_combo(self, combo: str) -> Set[keyboard.Key]:
        """Parse combo string like 'ctrl+alt' into pynput Key objects."""
        key_map = {
            "ctrl": keyboard.Key.ctrl,
            "ctrl_l": keyboard.Key.ctrl_l,
            "ctrl_r": keyboard.Key.ctrl_r,
            "alt": keyboard.Key.alt,
            "alt_l": keyboard.Key.alt_l,
            "alt_r": keyboard.Key.alt_r,
            "shift": keyboard.Key.shift,
            "shift_l": keyboard.Key.shift_l,
            "shift_r": keyboard.Key.shift_r,
            "cmd": keyboard.Key.cmd,
            "cmd_l": keyboard.Key.cmd_l,
            "cmd_r": keyboard.Key.cmd_r,
            "super": keyboard.Key.cmd,
        }

        keys = set()
        for part in combo.lower().split("+"):
            part = part.strip()
            if part in key_map:
                keys.add(key_map[part])
            else:
                logger.warning(f"Unknown key in combo: {part}")

        return keys

    def _is_active(self) -> bool:
        """Check if all target keys are currently pressed."""
        # For generic keys (ctrl, alt, shift), check if any variant is pressed
        for target in self._target_keys:
            if target in self._currently_pressed:
                continue
            # Check left/right variants
            if target == keyboard.Key.ctrl:
                if keyboard.Key.ctrl_l in self._currently_pressed or keyboard.Key.ctrl_r in self._currently_pressed:
                    continue
            elif target == keyboard.Key.alt:
                if keyboard.Key.alt_l in self._currently_pressed or keyboard.Key.alt_r in self._currently_pressed:
                    continue
            elif target == keyboard.Key.shift:
                if keyboard.Key.shift_l in self._currently_pressed or keyboard.Key.shift_r in self._currently_pressed:
                    continue
            elif target == keyboard.Key.cmd:
                if keyboard.Key.cmd_l in self._currently_pressed or keyboard.Key.cmd_r in self._currently_pressed:
                    continue
            return False
        return True

    def _on_press(self, key):
        with self._lock:
            if isinstance(key, keyboard.Key):
                self._currently_pressed.add(key)

            active = self._is_active()
            if active and not self._was_active:
                self._was_active = True
                logger.debug("Hotkey pressed")
                if self.on_transition:
                    self.on_transition(ShortcutTransition.PRESSED)

    def _on_release(self, key):
        with self._lock:
            if isinstance(key, keyboard.Key):
                self._currently_pressed.discard(key)
                # Also discard variants
                if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    self._currently_pressed.discard(keyboard.Key.ctrl)
                elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                    self._currently_pressed.discard(keyboard.Key.alt)
                elif key in (keyboard.Key.shift_l, keyboard.Key.shift_r):
                    self._currently_pressed.discard(keyboard.Key.shift)
                elif key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r):
                    self._currently_pressed.discard(keyboard.Key.cmd)

            active = self._is_active()
            if not active and self._was_active:
                self._was_active = False
                logger.debug("Hotkey released")
                if self.on_transition:
                    self.on_transition(ShortcutTransition.RELEASED)

    def start(self):
        """Start listening for the global hotkey."""
        if self._listener is not None:
            return

        logger.info(f"Starting global hotkey monitor: {self.combo}")
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        """Stop listening for the global hotkey."""
        if self._listener:
            self._listener.stop()
            self._listener = None
            logger.info("Global hotkey monitor stopped")