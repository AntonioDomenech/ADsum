from pathlib import Path

import numpy as np

from adsum.utils.audio import mix_audio_files, read_wave, write_wave


def test_mix_audio_files(tmp_path: Path) -> None:
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    tone_a = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    tone_b = np.sin(2 * np.pi * 880 * t).astype(np.float32)

    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    write_wave(path_a, tone_a, sample_rate)
    write_wave(path_b, tone_b, sample_rate)

    mix_path = tmp_path / "mix.wav"
    mix_audio_files([path_a, path_b], mix_path)

    mixed, sr = read_wave(mix_path)
    assert sr == sample_rate
    assert mixed.shape[0] == sample_rate
    assert mixed.shape[1] == 1


def test_mix_audio_files_resample(tmp_path: Path) -> None:
    sample_rate_a = 16000
    sample_rate_b = 22050
    t_a = np.linspace(0, 1, sample_rate_a, endpoint=False)
    t_b = np.linspace(0, 1, sample_rate_b, endpoint=False)
    tone_a = np.sin(2 * np.pi * 440 * t_a).astype(np.float32)
    tone_b = np.sin(2 * np.pi * 880 * t_b).astype(np.float32)

    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    write_wave(path_a, tone_a, sample_rate_a)
    write_wave(path_b, tone_b, sample_rate_b)

    mix_path = tmp_path / "mix.wav"
    mix_audio_files([path_a, path_b], mix_path)

    mixed, sr = read_wave(mix_path)
    assert sr == max(sample_rate_a, sample_rate_b)
    assert mixed.shape == (sr, 1)

    t = np.linspace(0, 1, sr, endpoint=False)
    tone_440 = np.sin(2 * np.pi * 440 * t)
    tone_880 = np.sin(2 * np.pi * 880 * t)
    correlation_440 = np.abs(np.dot(mixed[:, 0], tone_440) / sr)
    correlation_880 = np.abs(np.dot(mixed[:, 0], tone_880) / sr)

    assert correlation_440 > 0.1
    assert correlation_880 > 0.1
