"""System tray icon and control panel using PyQt6.

Mirrors MenuBarPanelManager.swift + CompanionPanelView.swift from the original macOS app.
"""

import logging
from PyQt6.QtWidgets import (
    QApplication,
    QSystemTrayIcon,
    QMenu,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont

from clicky.companion_manager import CompanionVoiceState
from typing import Optional

logger = logging.getLogger(__name__)


def create_default_icon() -> QIcon:
    """Create a simple blue circle icon for the system tray."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(37, 99, 235))  # #2563EB
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 24, 24)
    painter.end()
    return QIcon(pixmap)


class ControlPanel(QWidget):
    """Floating control panel that shows Clicky's status and controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(280)
        self.setFixedHeight(320)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Main container with dark background
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #101212;
                border-radius: 14px;
                border: 1px solid #1e2020;
            }
        """)
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(16, 16, 16, 16)
        container_layout.setSpacing(10)

        # Header: status dot + "Clicky"
        header_layout = QHBoxLayout()
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color: #4ade80; font-size: 14px;")
        header_layout.addWidget(self._status_dot)

        title = QLabel("Clicky")
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        container_layout.addLayout(header_layout)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #9ca3af; font-size: 13px;")
        container_layout.addWidget(self._status_label)

        # Hotkey hint
        hotkey_label = QLabel("Hold Ctrl+Alt to talk")
        hotkey_label.setStyleSheet("color: #6b7280; font-size: 12px;")
        container_layout.addWidget(hotkey_label)

        # Model picker
        model_label = QLabel("Model:")
        model_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
        container_layout.addWidget(model_label)

        self._model_combo = QComboBox()
        self._model_combo.addItems([
            "Claude Sonnet 4",
            "Claude Opus 4",
        ])
        self._model_combo.setStyleSheet("""
            QComboBox {
                background-color: #1e2020;
                color: white;
                border: 1px solid #2e3020;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        container_layout.addWidget(self._model_combo)

        # Spacer
        container_layout.addStretch()

        # Quit button
        quit_btn = QPushButton("Quit Clicky")
        quit_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ef4444;
            }
        """)
        quit_btn.clicked.connect(QApplication.quit)
        container_layout.addWidget(quit_btn)

        container.setLayout(container_layout)
        layout.addWidget(container)
        self.setLayout(layout)

    def update_state(self, state: CompanionVoiceState):
        """Update the panel display based on voice state."""
        state_map = {
            CompanionVoiceState.IDLE: ("Ready", "#4ade80"),
            CompanionVoiceState.LISTENING: ("Listening...", "#f59e0b"),
            CompanionVoiceState.PROCESSING: ("Thinking...", "#3b82f6"),
            CompanionVoiceState.RESPONDING: ("Speaking...", "#8b5cf6"),
        }
        text, color = state_map.get(state, ("Ready", "#4ade80"))
        self._status_label.setText(text)
        self._status_dot.setStyleSheet(f"color: {color}; font-size: 14px;")

    def get_selected_model(self) -> str:
        """Get the selected model identifier."""
        models = {
            0: "claude-sonnet-4-20250514",
            1: "claude-opus-4-20250514",
        }
        return models.get(self._model_combo.currentIndex(), "claude-sonnet-4-20250514")


class TrayIcon(QObject):
    """System tray icon with popup control panel."""

    toggle_panel = pyqtSignal()
    quit_app = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tray: Optional[QSystemTrayIcon] = None
        self._panel: Optional[ControlPanel] = None
        self._panel_visible = False

    def initialize(self, app: QApplication):
        """Set up the system tray icon and menu."""
        self._panel = ControlPanel()

        icon = create_default_icon()
        self._tray = QSystemTrayIcon(icon, app)

        # Context menu
        menu = QMenu()

        show_action = menu.addAction("Show Panel")
        show_action.triggered.connect(self._toggle_panel)

        menu.addSeparator()

        quit_action = menu.addAction("Quit Clicky")
        quit_action.triggered.connect(QApplication.quit)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_activated)
        self._tray.setToolTip("Clicky - AI Buddy")
        self._tray.show()

    def _on_activated(self, reason):
        """Handle tray icon clicks."""
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._toggle_panel()

    def _toggle_panel(self):
        """Show/hide the control panel."""
        if self._panel_visible:
            self._panel.hide()
            self._panel_visible = False
        else:
            # Position near the tray icon
            geo = self._tray.geometry()
            self._panel.move(geo.right() - self._panel.width(), geo.bottom() + 5)
            self._panel.show()
            self._panel.raise_()
            self._panel_visible = True

    def update_state(self, state: CompanionVoiceState):
        """Update panel and tray tooltip based on state."""
        if self._panel:
            self._panel.update_state(state)

        state_text = {
            CompanionVoiceState.IDLE: "Ready",
            CompanionVoiceState.LISTENING: "Listening...",
            CompanionVoiceState.PROCESSING: "Thinking...",
            CompanionVoiceState.RESPONDING: "Speaking...",
        }.get(state, "Ready")

        if self._tray:
            self._tray.setToolTip(f"Clicky - {state_text}")

    def get_selected_model(self) -> str:
        """Get the selected model from the control panel."""
        if self._panel:
            return self._panel.get_selected_model()
        return "claude-sonnet-4-20250514"

    def cleanup(self):
        """Clean up tray and panel."""
        if self._panel:
            self._panel.hide()
        if self._tray:
            self._tray.hide()