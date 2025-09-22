"""Audio processing utilities."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def read_wave(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels)
    else:
        data = data.reshape(-1, 1)
    data /= 32767.0
    return data, sample_rate


def write_wave(path: Path, data: np.ndarray, sample_rate: int) -> None:
    if data.ndim == 1:
        data = data[:, np.newaxis]
    data = np.clip(data, -1.0, 1.0)
    int16 = (data * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(data.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


def ensure_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1 or data.shape[1] == 1:
        return data.reshape(-1, 1)
    return data.mean(axis=1, keepdims=True)


def mix_audio_files(paths: Iterable[Path], output_path: Path) -> Path:
    data_arrays: List[np.ndarray] = []
    sample_rate: Optional[int] = None
    min_length: Optional[int] = None
    for path in paths:
        array, sr = read_wave(path)
        array = ensure_mono(array)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError("Sample rates must match for mixing")
        if min_length is None or array.shape[0] < min_length:
            min_length = array.shape[0]
        data_arrays.append(array)
    if not data_arrays:
        raise ValueError("No audio files supplied for mixing")
    assert min_length is not None and sample_rate is not None
    stacked = np.stack([arr[:min_length, 0] for arr in data_arrays], axis=0)
    mixed = stacked.mean(axis=0)
    write_wave(output_path, mixed.reshape(-1, 1), sample_rate)
    return output_path


__all__ = ["read_wave", "write_wave", "ensure_mono", "mix_audio_files"]

