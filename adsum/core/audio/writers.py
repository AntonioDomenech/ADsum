"""Utilities for writing PCM wave files."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable

import numpy as np


class AudioFileWriter:
    """Wave file writer that accepts floating point numpy arrays."""

    def __init__(self, path: Path, sample_rate: int, channels: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.channels = channels
        self._wave = wave.open(str(self.path), "wb")
        self._wave.setnchannels(channels)
        self._wave.setsampwidth(2)  # 16-bit PCM
        self._wave.setframerate(sample_rate)
        self._frames_written = 0

    def write(self, data: np.ndarray) -> None:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if data.shape[1] != self.channels:
            if data.shape[1] == 1 and self.channels == 2:
                data = np.repeat(data, 2, axis=1)
            else:
                raise ValueError("Channel mismatch when writing audio")
        clipped = np.clip(data, -1.0, 1.0)
        as_int16 = (clipped * 32767.0).astype(np.int16)
        self._wave.writeframes(as_int16.tobytes())
        self._frames_written += as_int16.shape[0]

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return self._frames_written / float(self.sample_rate)

    def close(self) -> None:
        self._wave.close()

    def __enter__(self) -> "AudioFileWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["AudioFileWriter"]

