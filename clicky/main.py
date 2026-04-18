"""Clicky - AI buddy that lives next to your cursor. Linux port.

Entry point: python -m clicky.main
"""

import sys
import logging
import signal

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer

from clicky.config import config
from clicky.companion_manager import CompanionManager, CompanionVoiceState
from clicky.overlay.cursor_overlay import OverlayManager
from clicky.tray.system_tray import TrayIcon


def setup_logging():
    """Configure logging for Clicky."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def check_config():
    """Validate configuration and warn about missing keys."""
    missing = config.validate()
    if missing:
        logging.warning(f"Missing configuration: {', '.join(missing)}")
        logging.warning("Some features may not work. Set these in .env or environment variables.")
    else:
        logging.info("All required configuration present")


def main():
    """Main entry point for Clicky."""
    setup_logging()
    logging.info("Starting Clicky...")

    # Validate config
    check_config()

    # Set high DPI scaling BEFORE creating QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Clicky")
    app.setQuitOnLastWindowClosed(False)  # Keep running in tray

    # Initialize overlay manager (transparent overlay windows on all screens)
    overlay = OverlayManager()
    overlay.initialize(app)

    # Initialize system tray icon
    tray = TrayIcon()
    tray.initialize(app)

    # Initialize companion manager (wires hotkey -> recorder -> transcription -> Claude -> TTS)
    manager = CompanionManager(
        overlay_manager=overlay,
        tray_icon=tray,
    )
    manager.initialize(app)

    # Connect tray model picker to manager
    def update_model():
        model = tray.get_selected_model()
        manager.set_model(model)

    # Connect quit signal
    def cleanup_and_quit():
        logging.info("Shutting down Clicky...")
        manager.cleanup()
        tray.cleanup()
        overlay.cleanup()
        app.quit()

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_quit())

    # Also handle Qt quit (from tray menu)
    app.aboutToQuit.connect(lambda: (
        manager.cleanup(),
        tray.cleanup(),
        overlay.cleanup(),
    ))

    logging.info("Clicky is running. Hold Ctrl+Alt to talk.")
    logging.info("Press Ctrl+C or use tray menu to quit.")

    # Run the Qt event loop
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        cleanup_and_quit()
        exit_code = 0

    sys.exit(exit_code)


if __name__ == "__main__":
    main()