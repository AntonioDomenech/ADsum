from __future__ import annotations

import io

import numpy as np
import pytest

import adsum.core.audio.ffmpeg_backend as ffmpeg_backend
from adsum.core.audio.base import CaptureInfo
from adsum.core.audio.ffmpeg_backend import (
    FFmpegCapture,
    parse_ffmpeg_device,
    CaptureError,
)


def test_parse_ffmpeg_device_accepts_overrides() -> None:
    spec = parse_ffmpeg_device(
        "pulse:bluez_source.test?sample_rate=48000&channels=2&sample_fmt=s16le&args=-thread_queue_size 1024",
        default_sample_rate=16000,
        default_channels=1,
    )

    assert spec.input_format == "pulse"
    assert spec.input_target == "bluez_source.test"
    assert spec.sample_rate == 48000
    assert spec.channels == 2
    assert spec.sample_format == "s16le"
    assert spec.args_before_input == ["-thread_queue_size", "1024"]


def test_parse_ffmpeg_device_rejects_unknown_option() -> None:
    with pytest.raises(CaptureError):
        parse_ffmpeg_device(
            "pulse:device?unexpected=1",
            default_sample_rate=16000,
            default_channels=1,
        )


def test_parse_ffmpeg_device_guesses_linux_defaults(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg_backend, "_detect_platform", lambda: "linux")

    spec = parse_ffmpeg_device(
        "default?channels=2",
        default_sample_rate=16000,
        default_channels=1,
    )

    assert spec.input_format == "pulse"
    assert spec.input_target == "default"
    assert spec.channels == 2


def test_parse_ffmpeg_device_guesses_windows_index(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg_backend, "_detect_platform", lambda: "windows")
    monkeypatch.setattr(
        ffmpeg_backend,
        "_lookup_sounddevice_device_name",
        lambda index: "USB Microphone" if index == 2 else None,
    )

    spec = parse_ffmpeg_device(
        "2",
        default_sample_rate=16000,
        default_channels=1,
    )

    assert spec.input_format == "dshow"
    assert spec.input_target == 'audio="USB Microphone"'


class _FakeProcess:
    def __init__(self, stdout_bytes: bytes) -> None:
        self.stdout = io.BufferedReader(io.BytesIO(stdout_bytes))
        self.stderr = io.BufferedReader(io.BytesIO(b""))
        self._returncode = 0

    def terminate(self) -> None:  # pragma: no cover - no behaviour change in tests
        pass

    def wait(self, timeout: float | None = None) -> int:  # pragma: no cover - trivial
        return self._returncode

    def poll(self) -> int:  # pragma: no cover - trivial
        return self._returncode

    def kill(self) -> None:  # pragma: no cover - trivial
        self._returncode = -9


def test_ffmpeg_capture_stream(monkeypatch) -> None:
    spec = parse_ffmpeg_device(
        "pulse:device?sample_rate=48000&channels=2&sample_fmt=s16le",
        default_sample_rate=16000,
        default_channels=1,
    )
    info = CaptureInfo(name="microphone", sample_rate=16000, channels=1)
    capture = FFmpegCapture(info, spec=spec, binary="ffmpeg", chunk_frames=2)

    sample = np.array([[0, 32767], [16384, -32768]], dtype=np.int16).tobytes()
    fake_process = _FakeProcess(sample)
    recorded: dict = {}

    monkeypatch.setattr(ffmpeg_backend, "_resolve_binary", lambda binary: "/usr/bin/ffmpeg")

    def fake_popen(cmd, stdin, stdout, stderr, bufsize):
        recorded["cmd"] = cmd
        return fake_process

    monkeypatch.setattr(ffmpeg_backend.subprocess, "Popen", fake_popen)

    capture.start()
    chunk = capture.read(timeout=1.0)
    capture.stop()
    capture.close()

    assert recorded["cmd"][0] == "/usr/bin/ffmpeg"
    assert chunk is not None
    assert chunk.shape == (2, 2)
    assert np.isclose(chunk[0, 1], 1.0, atol=1e-4)
    assert np.isclose(chunk[1, 0], 0.5, atol=1e-4)
