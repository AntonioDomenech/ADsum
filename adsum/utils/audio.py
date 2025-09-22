"""Audio processing utilities."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable, List, Tuple

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


def _resample_array(array: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return array
    mono = array[:, 0]
    length = mono.shape[0]
    if length == 0:
        return mono.reshape(0, 1)
    target_length = max(int(round(length * target_sr / sr)), 1)
    if target_length == 1:
        return np.full((1, 1), mono[0], dtype=array.dtype)
    original_positions = np.linspace(0, length - 1, num=length)
    target_positions = np.linspace(0, length - 1, num=target_length)
    resampled = np.interp(target_positions, original_positions, mono).astype(array.dtype, copy=False)
    return resampled.reshape(-1, 1)


def mix_audio_files(paths: Iterable[Path], output_path: Path) -> Path:
    data_entries: List[Tuple[np.ndarray, int]] = []
    for path in paths:
        array, sr = read_wave(path)
        array = ensure_mono(array)
        data_entries.append((array, sr))
    if not data_entries:
        raise ValueError("No audio files supplied for mixing")
    target_sample_rate = max(sr for _, sr in data_entries)
    resampled_arrays: List[np.ndarray] = []
    for array, sr in data_entries:
        resampled_arrays.append(_resample_array(array, sr, target_sample_rate))
    min_length = min(array.shape[0] for array in resampled_arrays)
    stacked = np.stack([arr[:min_length, 0] for arr in resampled_arrays], axis=0)
    mixed = stacked.mean(axis=0)
    write_wave(output_path, mixed.reshape(-1, 1), target_sample_rate)
    return output_path


__all__ = ["read_wave", "write_wave", "ensure_mono", "mix_audio_files"]

