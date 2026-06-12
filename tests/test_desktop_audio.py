from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from adsum.desktop.audio import measure_wave_file, mix_wave_files


def _write_tone(path: Path, *, sample_rate: int, seconds: float, frequency: float) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = 0.25 * np.sin(2 * math.pi * frequency * t)
    pcm = (tone * 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())


def test_mix_wave_files_creates_transcription_ready_mono_track(tmp_path: Path) -> None:
    mic = tmp_path / "mic.wav"
    system = tmp_path / "system.wav"
    mixed = tmp_path / "mixed.wav"

    _write_tone(mic, sample_rate=44100, seconds=1.0, frequency=440)
    _write_tone(system, sample_rate=48000, seconds=1.0, frequency=660)

    mix_wave_files([mic, system], mixed)

    metrics = measure_wave_file(mixed)
    assert mixed.exists()
    assert metrics.duration_seconds == 1.0
    assert metrics.peak > 0
    assert metrics.rms > 0

    with wave.open(str(mixed), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getframerate() == 16000


def test_measure_wave_file_handles_missing_path(tmp_path: Path) -> None:
    metrics = measure_wave_file(tmp_path / "missing.wav")

    assert metrics.duration_seconds == 0
    assert metrics.peak == 0
    assert metrics.rms == 0
