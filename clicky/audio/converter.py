"""Audio conversion utilities.

Converts between audio formats and builds WAV payloads.
Mirrors BuddyAudioConversionSupport.swift from the original macOS app.
"""

import struct
import io
import logging

import numpy as np
from typing import List

logger = logging.getLogger(__name__)


def pcm16_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Build a WAV file from raw PCM16 data.

    Args:
        pcm_data: Raw 16-bit PCM audio bytes
        sample_rate: Sample rate in Hz
        channels: Number of channels (1=mono, 2=stereo)

    Returns:
        Complete WAV file as bytes
    """
    num_samples = len(pcm_data) // 2  # 16-bit = 2 bytes per sample
    bytes_per_sample = 2
    block_align = channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    data_size = num_samples * bytes_per_sample

    # RIFF header
    wav = io.BytesIO()
    wav.write(b"RIFF")
    wav.write(struct.pack("<I", 36 + data_size))  # File size - 8
    wav.write(b"WAVE")

    # fmt chunk
    wav.write(b"fmt ")
    wav.write(struct.pack("<I", 16))  # Chunk size
    wav.write(struct.pack("<H", 1))   # PCM format
    wav.write(struct.pack("<H", channels))
    wav.write(struct.pack("<I", sample_rate))
    wav.write(struct.pack("<I", byte_rate))
    wav.write(struct.pack("<H", block_align))
    wav.write(struct.pack("<H", 16))  # Bits per sample

    # data chunk
    wav.write(b"data")
    wav.write(struct.pack("<I", data_size))
    wav.write(pcm_data)

    return wav.getvalue()


def resample_audio(pcm_data: bytes, orig_rate: int, target_rate: int) -> bytes:
    """Simple audio resampling using linear interpolation.

    Args:
        pcm_data: Raw PCM16 bytes at orig_rate
        orig_rate: Original sample rate
        target_rate: Target sample rate

    Returns:
        PCM16 bytes at target_rate
    """
    if orig_rate == target_rate:
        return pcm_data

    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    ratio = target_rate / orig_rate
    new_length = int(len(samples) * ratio)

    indices = np.linspace(0, len(samples) - 1, new_length)
    resampled = np.interp(indices, np.arange(len(samples)), samples)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

    return resampled.tobytes()


def normalize_audio_level(pcm_data: bytes, chunk_size: int = 1024) -> List[float]:
    """Calculate audio levels for waveform visualization.

    Args:
        pcm_data: Raw PCM16 bytes
        chunk_size: Number of samples per chunk

    Returns:
        List of RMS levels (0.0-1.0) per chunk
    """
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    levels = []
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
        levels.append(min(rms * 5.0, 1.0))
    return levels