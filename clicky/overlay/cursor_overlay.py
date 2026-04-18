"""Transparent cursor overlay window using PyQt6.

Creates a full-screen transparent overlay on each monitor that shows:
- Blue triangle cursor that follows the mouse
- Cursor flying animation to pointed elements
- Waveform when listening
- Spinner when processing
- Streaming response text

Mirrors OverlayWindow.swift from the original macOS app.
"""

import math
import logging
import os
from typing import Callable, Optional, List

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
)
from PyQt6.QtGui import QScreen
from PyQt6.QtCore import (
    Qt,
    QTimer,
    QPropertyAnimation,
    QPoint,
    QPointF,
    pyqtSignal,
    pyqtProperty,
    pyqtSlot,
    QMetaObject,
    Q_ARG,
)
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QPolygonF,
    QFont,
    QCursor,
)

# Xlib-based cursor control via ctypes — no persistent listener threads
# that could interfere with Qt's event processing.
# Falls back to pynput.mouse if XTest is unavailable.
try:
    import ctypes
    from ctypes import cdll, c_int, c_uint, c_void_p, byref
    _xlib = cdll.LoadLibrary("libX11.so.6")
    _xlib.XOpenDisplay.restype = c_void_p
    _xlib.XOpenDisplay.argtypes = [c_void_p]
    _xlib.XCloseDisplay.argtypes = [c_void_p]
    _xlib.XWarpPointer.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_uint, c_int, c_int]
    _xlib.XFlush.argtypes = [c_void_p]
    # XTest extension for simulated clicks
    _xtest = cdll.LoadLibrary("libXtst.so.6")
    _xtest.XTestFakeButtonEvent.argtypes = [c_void_p, c_int, c_int, c_void_p]
    _xtest.XFlush.argtypes = [c_void_p]
    _xlib_available = True
    _xtest_available = True
except (OSError, AttributeError) as e:
    _xlib = None
    _xtest = None
    _xtest_available = False
    _xlib_available = False
    logger.warning(f"Xlib not available for cursor control: {e}")

# pynput fallback for click simulation when XTest is unavailable
try:
    from pynput.mouse import Controller as MouseController
    _pynput_mouse = MouseController()
    _pynput_available = True
except Exception:
    _pynput_mouse = None
    _pynput_available = False
    logger.warning("pynput mouse not available for click simulation")

logger = logging.getLogger(__name__)


class BlueCursorWidget(QWidget):
    """Transparent overlay showing the blue Clicky cursor.

    The cursor is a blue triangle that follows the mouse and can
    fly to target coordinates with a bezier animation.
    """

    # Signals
    fly_requested = pyqtSignal(float, float, str)  # x, y, label — thread-safe
    animation_finished = pyqtSignal()

    def __init__(self, screen_geometry, parent=None):
        super().__init__(parent)
        self.screen_geometry = screen_geometry

        # Window flags for frameless, transparent, always-on-top overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # Don't show in taskbar
            | Qt.WindowType.X11BypassWindowManagerHint  # Bypass WM on X11
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)  # Click-through
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Cursor state
        self._cursor_x = 0.0
        self._cursor_y = 0.0
        self._target_x = 0.0
        self._target_y = 0.0
        self._cursor_visible = False
        self._cursor_label = ""

        # Animation
        self._anim_progress = 0.0
        self._anim_start_x = 0.0
        self._anim_start_y = 0.0
        self._anim_control_x = 0.0
        self._anim_control_y = 0.0
        self._is_animating = False
        # True after animation lands — prevents _mouse_timer from snapping cursor away
        self._landed_at_target = False
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(16)  # ~60fps
        self._anim_timer.timeout.connect(self._animate_step)

        # Thread-safe fly signal -> slot connection (marshals to GUI thread)
        self.fly_requested.connect(
            self.fly_to_point, Qt.ConnectionType.QueuedConnection
        )

        # Waveform
        self._audio_level = 0.0

        # Response text
        self._response_text = ""
        self._is_processing = False

        # Set geometry to cover the screen
        self.setGeometry(screen_geometry)

        # Track mouse position
        self._mouse_timer = QTimer()
        self._mouse_timer.setInterval(50)
        self._mouse_timer.timeout.connect(self._update_mouse_position)
        self._mouse_timer.start()

    def showEvent(self, event):
        """Re-apply click-through every time the overlay is shown (X11 quirk)."""
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        super().showEvent(event)

    def _update_mouse_position(self):
        """Update cursor position to follow the mouse."""
        # Don't override if animating or if we just landed at a target point
        if (not self._is_animating and self._cursor_visible
                and not self._landed_at_target):
            pos = QCursor.pos()
            self._cursor_x = pos.x() - self.screen_geometry.x()
            self._cursor_y = pos.y() - self.screen_geometry.y()
            self.update()

    @pyqtSlot()
    def show_cursor(self):
        """Show the blue cursor at its current (or mouse) position."""
        self._landed_at_target = False  # Always re-enable real mouse tracking
        # Only snap to mouse if not animating
        if not self._is_animating:
            pos = QCursor.pos()
            self._cursor_x = pos.x() - self.screen_geometry.x()
            self._cursor_y = pos.y() - self.screen_geometry.y()
        self._cursor_visible = True
        self.show()
        self.update()

    @pyqtSlot()
    def release_cursor_lock(self):
        """Release the cursor lock from fly_to_point, allowing normal mouse tracking.

        Unlike show_cursor(), this does NOT snap to the current mouse position —
        the cursor gradually catches up via _update_mouse_position timer.
        Use this when TTS starts to prevent cursor from being stuck at a target.
        """
        self._landed_at_target = False

    @pyqtSlot()
    def hide_cursor(self):
        """Hide the blue cursor."""
        self._cursor_visible = False
        self.hide()

    @pyqtSlot(float, float, str)
    def fly_to_point(self, x: float, y: float, label: str = ""):
        """Animate the cursor flying to a target point.

        Args:
            x: Target X coordinate (screen-absolute)
            y: Target Y coordinate (screen-absolute)
            label: Label to show at the target
        """
        logger.info(f"BlueCursorWidget.fly_to_point called: ({x}, {y}) label='{label}'")
        logger.info(f"  screen_geometry: x()={self.screen_geometry.x()} y()={self.screen_geometry.y()} w={self.screen_geometry.width()} h={self.screen_geometry.height()}")
        self._target_x = x - self.screen_geometry.x()
        self._target_y = y - self.screen_geometry.y()
        logger.info(f"  widget-relative target: ({self._target_x:.1f}, {self._target_y:.1f})")
        self._cursor_label = label

        # Start animation
        self._anim_start_x = self._cursor_x
        self._anim_start_y = self._cursor_y

        # Control point for bezier arc (offset upward)
        mid_x = (self._anim_start_x + self._target_x) / 2
        mid_y = min(self._anim_start_y, self._target_y) - 100
        self._anim_control_x = mid_x
        self._anim_control_y = mid_y

        # Lock cursor to animation start — prevents _mouse_timer from snapping cursor
        # back to real mouse mid-flight. We keep it locked at the end too so the
        # cursor stays at the target after landing (until user moves or next animation).
        self._landed_at_target = True
        logger.info(f"Starting animation: from ({self._cursor_x:.1f}, {self._cursor_y:.1f}) to ({self._target_x:.1f}, {self._target_y:.1f})")
        self._anim_progress = 0.0
        self._is_animating = True
        self._anim_timer.start()

    def _animate_step(self):
        """One step of the fly-to animation."""
        if not self._is_animating:
            self._anim_timer.stop()
            return

        self._anim_progress += 0.05  # ~20 frames

        if self._anim_progress >= 1.0:
            self._anim_progress = 1.0
            self._is_animating = False
            self._landed_at_target = True  # Lock cursor at target, stop _mouse_timer override
            self._anim_timer.stop()
            self._cursor_x = self._target_x
            self._cursor_y = self._target_y
            # Move the real system cursor to the target point
            logger.info(f"Animation landed at ({self._target_x}, {self._target_y}), calling _move_system_cursor")
            self._move_system_cursor(int(self._target_x), int(self._target_y))
            self.animation_finished.emit()
        else:
            # Quadratic bezier
            t = self._anim_progress
            # Ease-in-out
            t = t * t * (3 - 2 * t)

            self._cursor_x = (
                (1 - t) ** 2 * self._anim_start_x
                + 2 * (1 - t) * t * self._anim_control_x
                + t**2 * self._target_x
            )
            self._cursor_y = (
                (1 - t) ** 2 * self._anim_start_y
                + 2 * (1 - t) * t * self._anim_control_y
                + t**2 * self._target_y
            )

        self.update()

    @pyqtSlot(float)
    def set_audio_level(self, level: float):
        """Set the audio level for waveform display (0.0-1.0)."""
        self._audio_level = level
        if self._cursor_visible:
            self.update()

    def _move_system_cursor(self, x: int, y: int):
        """Move the real system cursor to (x, y) and click."""
        logger.info(f"_move_system_cursor({x}, {y}) xlib={_xlib_available} xtest={_xtest_available} pynput={_pynput_available}")
        abs_x = x + self.screen_geometry.x()
        abs_y = y + self.screen_geometry.y()
        logger.info(f"  absolute coords: ({abs_x}, {abs_y})")

        succeeded = False

        # Try Xlib + XTest first (most reliable on X11)
        if _xlib_available:
            try:
                display = _xlib.XOpenDisplay(None)
                if not display:
                    logger.warning("_move_system_cursor: XOpenDisplay returned NULL")
                else:
                    _xlib.XWarpPointer(display, 0, 0, 0, 0, 0, 0, abs_x, abs_y)
                    _xlib.XFlush(display)
                    logger.info(f"_move_system_cursor: warped to ({abs_x}, {abs_y})")
                    if _xtest_available and _xtest:
                        _xtest.XTestFakeButtonEvent(display, 1, 1, 0)  # press
                        _xtest.XFlush(display)
                        _xtest.XTestFakeButtonEvent(display, 1, 0, 0)  # release
                        _xtest.XFlush(display)
                        logger.info(f"_move_system_cursor: clicked via XTest at ({abs_x}, {abs_y})")
                        succeeded = True
                    _xlib.XCloseDisplay(display)
            except Exception as e:
                logger.error(f"_move_system_cursor Xlib failed: {e}")

        # Fallback: pynput mouse (works if DISPLAY is set and user has X permissions)
        if not succeeded and _pynput_available and _pynput_mouse is not None:
            try:
                _pynput_mouse.position = (abs_x, abs_y)
                _pynput_mouse.press(_pynput_mouse.Button.left)
                _pynput_mouse.release(_pynput_mouse.Button.left)
                logger.info(f"_move_system_cursor: clicked via pynput at ({abs_x}, {abs_y})")
                succeeded = True
            except Exception as e:
                logger.error(f"_move_system_cursor pynput failed: {e}")

        if not succeeded:
            logger.warning(f"_move_system_cursor: all backends failed — DISPLAY={os.environ.get('DISPLAY','(not set)')}, xlib={_xlib_available}, xtest={_xtest_available}, pynput={_pynput_available}")

    @pyqtSlot(str)
    def set_response_text(self, text: str):
        """Set the response text to display."""
        self._response_text = text
        if self._cursor_visible:
            self.update()

    @pyqtSlot(bool)
    def set_processing(self, processing: bool):
        """Show/hide processing spinner."""
        self._is_processing = processing
        if self._cursor_visible:
            self.update()

    def paintEvent(self, event):
        """Paint the overlay."""
        if not self._cursor_visible:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw blue cursor triangle
        self._draw_cursor(painter)

        # Draw waveform (when listening)
        if self._audio_level > 0.01:
            self._draw_waveform(painter)

        # Draw processing spinner
        if self._is_processing:
            self._draw_spinner(painter)

        painter.end()

    def _draw_cursor(self, painter: QPainter):
        """Draw the blue triangle cursor."""
        cx, cy = self._cursor_x, self._cursor_y
        size = 20

        # Blue triangle pointing right-down
        triangle = QPolygonF([
            QPointF(cx, cy),
            QPointF(cx, cy + size),
            QPointF(cx + size * 0.7, cy + size * 0.5),
        ])

        # Glow effect
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(37, 99, 235, 60)))  # Semi-transparent blue
        glow_triangle = QPolygonF([
            QPointF(cx - 4, cy - 4),
            QPointF(cx - 4, cy + size + 4),
            QPointF(cx + size * 0.7 + 6, cy + size * 0.5),
        ])
        painter.drawPolygon(glow_triangle)

        # Main cursor
        painter.setBrush(QBrush(QColor(37, 99, 235)))  # #2563EB
        painter.drawPolygon(triangle)

        # Border
        painter.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(triangle)

    def _draw_waveform(self, painter: QPainter):
        """Draw a simple audio waveform near the cursor."""
        cx, cy = self._cursor_x, self._cursor_y
        level = self._audio_level

        painter.setPen(QPen(QColor(37, 99, 235, 180), 2))
        num_bars = 5
        bar_width = 3
        spacing = 6
        start_x = cx + 30

        for i in range(num_bars):
            # Vary height based on level and position
            variance = 1.0 - abs(i - num_bars / 2) / (num_bars / 2)
            height = max(4, int(level * 20 * variance))
            x = start_x + i * spacing
            painter.drawRect(int(x), int(cy - height / 2), bar_width, height)

    def _draw_spinner(self, painter: QPainter):
        """Draw a processing spinner near the cursor."""
        cx, cy = self._cursor_x, self._cursor_y
        spinner_x = cx + 30
        spinner_y = cy + 5
        radius = 8

        painter.setPen(QPen(QColor(37, 99, 235), 2.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(
            int(spinner_x - radius),
            int(spinner_y - radius),
            int(radius * 2),
            int(radius * 2),
            0,  # startAngle
            270 * 16,  # spanAngle (3/4 circle)
        )

    def _draw_response_text(self, painter: QPainter):
        """Draw response text near the cursor."""
        cx, cy = self._cursor_x, self._cursor_y

        # Text bubble
        font = QFont("Inter", 13)
        painter.setFont(font)

        text = self._response_text
        if len(text) > 200:
            text = text[:200] + "..."

        # Measure text
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        padding = 12

        # Position: to the right of cursor, offset down
        bubble_x = cx + 30
        bubble_y = cy + 25
        bubble_w = text_width + padding * 2
        bubble_h = text_height + padding * 2

        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(16, 18, 17, 220)))  # Dark background
        painter.drawRoundedRect(
            int(bubble_x), int(bubble_y),
            int(bubble_w), int(bubble_h),
            10, 10,
        )

        # Text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            int(bubble_x + padding),
            int(bubble_y + padding + metrics.ascent()),
            text,
        )

    def _draw_label(self, painter: QPainter):
        """Draw a point label at the target location."""
        x = self._target_x
        y = self._target_y - 30  # Above the point

        font = QFont("Inter", 11)
        painter.setFont(font)

        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self._cursor_label)
        text_height = metrics.height()
        padding = 6

        # Background pill
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(37, 99, 235, 200)))
        painter.drawRoundedRect(
            int(x - text_width / 2 - padding),
            int(y - text_height / 2 - padding),
            int(text_width + padding * 2),
            int(text_height + padding * 2),
            8, 8,
        )

        # Text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            int(x - text_width / 2),
            int(y + metrics.ascent() / 2),
            self._cursor_label,
        )


class OverlayManager:
    """Manages overlay windows across all screens."""

    def __init__(self):
        self._overlays: List[BlueCursorWidget] = []
        self._app: Optional[QApplication] = None
        # Scale factors for coordinate conversion: LLM returns image-space coords,
        # we scale them by display/screenshot ratio to get display-space coords.
        # Keyed by screen index.
        self._scale_factors: dict[int, tuple[float, float]] = {}

    def initialize(self, app: QApplication):
        """Initialize overlays for all screens."""
        self._app = app
        self._create_overlays()

    def _create_overlays(self):
        """Create an overlay widget for each screen."""
        self._overlays.clear()
        screens = QApplication.screens()
        for screen in screens:
            geometry = screen.geometry()
            overlay = BlueCursorWidget(geometry)
            self._overlays.append(overlay)
            logger.info(f"Overlay created: screen geometry={geometry.width()}x{geometry.height()} at ({geometry.x()},{geometry.y()}), "
                        f"devicePixelRatio={screen.devicePixelRatio()}")

        logger.info(f"Created {len(self._overlays)} overlay windows")

    def show_cursor(self):
        """Show cursor on all overlays (thread-safe)."""
        for overlay in self._overlays:
            # Marshal to Qt main thread via queued invocation
            QMetaObject.invokeMethod(
                overlay, "show_cursor",
                Qt.ConnectionType.QueuedConnection
            )

    def hide_cursor(self):
        """Hide cursor on all overlays (thread-safe)."""
        for overlay in self._overlays:
            QMetaObject.invokeMethod(
                overlay, "hide_cursor",
                Qt.ConnectionType.QueuedConnection
            )

    def set_scale_factors(self, screen_idx: int, display_width: int, display_height: int, screenshot_width: int, screenshot_height: int):
        """Update scale factors for coordinate conversion (image space -> display space).

        Called from companion_manager after each screen capture so fly_to_point
        can scale LLM-returned image-space coordinates to display-space.

        Note: We prefer the overlay's own screen geometry over mss-reported display
        dimensions, since mss can report incorrect virtual desktop sizes on some
        systems (e.g., reporting 2560x1440 when the actual display is 2200x1650).
        """
        # Store display dimensions for normalized-to-display coordinate conversion
        # (The LLM now returns normalized 0.0-1.0 coords, multiplied by display dim to get pixels)
        self._display_dims: dict[int, tuple[int, int]] = {}
        if screenshot_width > 0 and screenshot_height > 0:
            overlay_display_w = display_width
            overlay_display_h = display_height
            if screen_idx < len(self._overlays):
                geo = self._overlays[screen_idx].screen_geometry
                if geo.width() > 0 and geo.height() > 0:
                    overlay_display_w = geo.width()
                    overlay_display_h = geo.height()
            self._display_dims[screen_idx] = (overlay_display_w, overlay_display_h)
            logger.info(f"Display dims for screen {screen_idx}: {overlay_display_w}x{overlay_display_h} "
                        f"(overlay_geo={overlay_display_w}x{overlay_display_h}, screenshot={screenshot_width}x{screenshot_height})")

    def fly_to_point(self, x: float, y: float, label: str = "", screen: int = 1):
        """Fly cursor to a point on the specified screen.

        Args:
            x: X coordinate — either normalized (0.0-1.0) or pixel value
            y: Y coordinate — either normalized (0.0-1.0) or pixel value
            label: Label to show at the point
            screen: 1-based screen number

        Note: The LLM now returns normalized coordinates (0.0-1.0) relative to the
        full screenshot bounds. We multiply by the actual display dimensions to get
        pixel coordinates.
        """
        screen_idx = max(0, min(screen - 1, len(self._overlays) - 1))
        display_w, display_h = self._display_dims.get(screen_idx, (1920, 1080))

        # Auto-detect: if x > 1.0, treat as pixel coordinate (backward compat)
        # if 0.0 <= x <= 1.0, treat as normalized
        if x > 1.0 or y > 1.0:
            # Pixel coordinates — use directly (but cap to display bounds)
            x_display = max(0, min(int(x), display_w))
            y_display = max(0, min(int(y), display_h))
            logger.info(f"fly_to_point: pixel coords ({x}, {y}) -> display ({x_display}, {y_display})")
        else:
            # Normalized 0-1 coords — multiply by display dimensions
            x_display = int(x * display_w)
            y_display = int(y * display_h)
            logger.info(f"fly_to_point: normalized ({x:.3f}, {y:.3f}) -> display ({x_display}, {y_display})")

        overlay = self._overlays[screen_idx]

        # Convert display-space to widget-relative
        geo = overlay.screen_geometry
        abs_x = geo.x() + x_display
        abs_y = geo.y() + y_display

        # Emit thread-safe signal — Qt queued connection handles GUI thread marshalling
        logger.info(f"OverlayManager.fly_to_point: emitting fly_requested for ({abs_x}, {abs_y}) label='{label}'")
        overlay.fly_requested.emit(abs_x, abs_y, label)

    def set_audio_level(self, level: float):
        """Set audio level on all overlays (thread-safe)."""
        for overlay in self._overlays:
            # Marshal to Qt main thread — set_audio_level is a @pyqtSlot(float)
            QMetaObject.invokeMethod(
                overlay, "set_audio_level",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(float, float(level)),
            )

    def set_response_text(self, text: str):
        """Set response text on all overlays (thread-safe)."""
        for overlay in self._overlays:
            # Marshal to Qt main thread — set_response_text is a @pyqtSlot(str)
            QMetaObject.invokeMethod(
                overlay, "set_response_text",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, text),
            )

    def set_processing(self, processing: bool):
        """Set processing state on all overlays (thread-safe)."""
        for overlay in self._overlays:
            # Marshal to Qt main thread — set_processing is a @pyqtSlot(bool)
            QMetaObject.invokeMethod(
                overlay, "set_processing",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(bool, bool(processing)),
            )

    def release_cursor_lock(self):
        """Release cursor lock from fly_to_point on all overlays (thread-safe).

        Unlike show_cursor(), this does NOT snap to the current mouse position —
        the cursor gradually catches up via _update_mouse_position timer.
        Call this when TTS starts to prevent cursor from being stuck at a target.
        """
        for overlay in self._overlays:
            QMetaObject.invokeMethod(
                overlay, "release_cursor_lock",
                Qt.ConnectionType.QueuedConnection,
            )

    def cleanup(self):
        """Clean up overlay windows."""
        for overlay in self._overlays:
            overlay.hide()
            overlay.deleteLater()
        self._overlays.clear()