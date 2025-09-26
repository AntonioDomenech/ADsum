"""Audio processing utilities."""

from __future__ import annotations

import wave
from dataclasses import dataclass
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


@dataclass(slots=True)
class AudioChunk:
    """Metadata about an intermediate audio chunk produced during splitting."""

    path: Path
    start: float
    duration: float


def split_wave_file(path: Path, max_bytes: int) -> List[AudioChunk]:
    """Split a WAV file into sequential chunks that do not exceed ``max_bytes``.

    The function streams frames from ``path`` without loading the full file into
    memory. Each chunk is written as a temporary WAV file in the same directory
    and the resulting metadata captures the chunk's start offset and duration so
    downstream consumers can adjust timestamps when merging responses.
    """

    if max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    chunks: List[AudioChunk] = []
    with wave.open(str(path), "rb") as source:
        n_channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frame_rate = source.getframerate()
        frame_size = n_channels * sample_width
        if frame_size <= 0:
            raise ValueError("Invalid WAV metadata; frame size must be positive")

        # Account for WAV header overhead to keep the final file size under the
        # limit. A standard PCM header is 44 bytes.
        payload_budget = max_bytes - 44
        if payload_budget <= 0:
            raise ValueError("max_bytes is too small to fit a WAV header")

        max_frames = max(payload_budget // frame_size, 1)
        start_time = 0.0
        index = 0

        while True:
            frames = source.readframes(max_frames)
            if not frames:
                break

            frame_count = len(frames) // frame_size
            duration = frame_count / frame_rate if frame_rate else 0.0

            chunk_path = path.with_name(f"{path.stem}.chunk{index:03d}{path.suffix}")
            if chunk_path.exists():
                chunk_path.unlink()

            with wave.open(str(chunk_path), "wb") as chunk_file:
                chunk_file.setnchannels(n_channels)
                chunk_file.setsampwidth(sample_width)
                chunk_file.setframerate(frame_rate)
                chunk_file.writeframes(frames)

            if chunk_path.stat().st_size > max_bytes:
                chunk_path.unlink(missing_ok=True)
                raise ValueError("Chunk size exceeded max_bytes budget")

            chunks.append(AudioChunk(path=chunk_path, start=start_time, duration=duration))
            start_time += duration
            index += 1

    return chunks


__all__ = [
    "read_wave",
    "write_wave",
    "ensure_mono",
    "mix_audio_files",
    "AudioChunk",
    "split_wave_file",
]

