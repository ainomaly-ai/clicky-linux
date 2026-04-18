from setuptools import setup, find_packages

setup(
    name="clicky-linux",
    version="0.1.0",
    description="AI buddy that lives next to your cursor - Linux port",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "PyQt6>=6.6.0",
        "pynput>=1.7.6",
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "httpx>=0.27.0",
        "websockets>=12.0",
        "mss>=9.0.0",
        "Pillow>=10.0.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.29.0",
        "pyttsx3>=2.90",
        "python-dotenv>=1.0.0",
        "pydub>=0.25.1",
        "edge-tts>=6.1.0",
        "faster-whisper>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "clicky=clicky.main:main",
            "clicky-proxy=clicky.proxy.server:run_proxy",
        ],
    },
)