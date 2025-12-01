from __future__ import annotations

import types

import sys
import types

import pytest

import adsum.core.audio.factory as factory_module
import adsum.core.audio.ffmpeg_backend as ffmpeg_backend
from adsum.core.audio.factory import CaptureConfigurationError, CaptureRequest, create_capture
from adsum.core.audio.base import CaptureError


class _DummySettings:
    audio_backend = "ffmpeg"
    ffmpeg_binary = "ffmpeg"
    chunk_seconds = 0.5


def test_create_capture_rejects_unsupported_backend() -> None:
    """Only the FFmpeg backend should be accepted for capture creation."""

    request = CaptureRequest(
        channel="microphone",
        device="default",
        sample_rate=16_000,
        channels=1,
        backend="sounddevice",
    )

    with pytest.raises(CaptureConfigurationError) as excinfo:
        create_capture(request)

    assert "Only 'ffmpeg' is supported" in str(excinfo.value)


def test_create_capture_raises_when_ffmpeg_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing FFmpeg binaries should surface a configuration error."""

    monkeypatch.setattr(factory_module, "get_settings", lambda: _DummySettings())
    monkeypatch.setattr(ffmpeg_backend, "_resolve_binary", lambda binary: None)

    request = CaptureRequest(
        channel="microphone",
        device="pulse:default",
        sample_rate=16_000,
        channels=1,
        backend="ffmpeg",
    )

    with pytest.raises(CaptureConfigurationError) as excinfo:
        create_capture(request)

    message = str(excinfo.value)
    assert "FFmpeg binary" in message
    assert "Install FFmpeg" in message


def test_create_capture_returns_ffmpeg_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful capture creation should instantiate the FFmpeg backend."""

    monkeypatch.setattr(factory_module, "get_settings", lambda: _DummySettings())
    monkeypatch.setattr(ffmpeg_backend, "_resolve_binary", lambda binary: "/usr/bin/ffmpeg")

    captured: dict[str, object] = {}

    class DummyCapture:
        def __init__(self, info, spec, binary, chunk_frames) -> None:  # pragma: no cover - simple init
            captured["info"] = info
            captured["spec"] = spec
            captured["binary"] = binary
            captured["chunk_frames"] = chunk_frames

    fake_module = types.SimpleNamespace(
        FFmpegBinaryNotFoundError=ffmpeg_backend.FFmpegBinaryNotFoundError,
        FFmpegCapture=DummyCapture,
        _resolve_binary=ffmpeg_backend._resolve_binary,
        parse_ffmpeg_device=ffmpeg_backend.parse_ffmpeg_device,
    )

    monkeypatch.setitem(sys.modules, "adsum.core.audio.ffmpeg_backend", fake_module)

    request = CaptureRequest(
        channel="microphone",
        device="pulse:default?sample_rate=16000&channels=1",
        sample_rate=16_000,
        channels=1,
        backend="ffmpeg",
    )

    capture = create_capture(request)

    assert isinstance(capture, DummyCapture)
    assert captured["binary"] == "/usr/bin/ffmpeg"
    assert captured["chunk_frames"] == int(16_000 * _DummySettings().chunk_seconds)
    assert captured["spec"].input_format == "pulse"
    assert captured["spec"].input_target == "default"


def test_create_capture_prefers_soundcard_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """WASAPI loopback requests should prioritise the soundcard backend."""

    fake_settings = _DummySettings()
    monkeypatch.setattr(factory_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(ffmpeg_backend, "_resolve_binary", lambda binary: "/usr/bin/ffmpeg")

    fake_spec = ffmpeg_backend.FFmpegDeviceSpec(
        input_format="wasapi",
        input_target='"Speakers (Realtek(R) Audio)"',
        args_before_input=[],
        input_options=["-loopback", "1"],
        args_after_input=[],
        sample_rate=48_000,
        channels=2,
        sample_format="f32le",
        chunk_frames=None,
    )
    monkeypatch.setattr(ffmpeg_backend, "parse_ffmpeg_device", lambda *_, **__: fake_spec)

    captured_kwargs: dict = {}

    def fake_soundcard_capture(**kwargs):
        captured_kwargs["kwargs"] = kwargs
        return "soundcard"

    soundcard_module = types.SimpleNamespace(SoundCardLoopbackCapture=fake_soundcard_capture)
    monkeypatch.setitem(sys.modules, "adsum.core.audio.soundcard_backend", soundcard_module)

    # Ensure sounddevice would raise if unexpectedly invoked.
    def _unexpected_sounddevice(**_kwargs):
        raise AssertionError("sounddevice backend should not be used when soundcard succeeds")

    sounddevice_module = types.SimpleNamespace(SoundDeviceLoopbackCapture=_unexpected_sounddevice)
    monkeypatch.setitem(sys.modules, "adsum.core.audio.sounddevice_backend", sounddevice_module)

    request = CaptureRequest(
        channel="system",
        device='wasapi:"Speakers (Realtek(R) Audio)"?loopback=1',
        sample_rate=16_000,
        channels=1,
        backend="ffmpeg",
        chunk_seconds=0.1,
    )

    capture = create_capture(request)

    assert capture == "soundcard"
    assert captured_kwargs["kwargs"]["target"] == "Speakers (Realtek(R) Audio)"
    assert captured_kwargs["kwargs"]["sample_rate"] == 48_000
    assert captured_kwargs["kwargs"]["channels"] == 2


def test_create_capture_prefers_sounddevice_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """WASAPI loopback requests should try the sounddevice backend."""

    fake_settings = _DummySettings()
    monkeypatch.setattr(factory_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(ffmpeg_backend, "_resolve_binary", lambda binary: "/usr/bin/ffmpeg")

    fake_spec = ffmpeg_backend.FFmpegDeviceSpec(
        input_format="wasapi",
        input_target="Speakers (Realtek(R) Audio)",
        args_before_input=[],
        input_options=["-loopback", "1"],
        args_after_input=[],
        sample_rate=48_000,
        channels=2,
        sample_format="f32le",
        chunk_frames=None,
    )

    monkeypatch.setattr(ffmpeg_backend, "parse_ffmpeg_device", lambda *_, **__: fake_spec)

    captured_kwargs: dict = {}

    import types

    def fake_sounddevice_capture(**kwargs):
        captured_kwargs["capture"] = kwargs
        return "sounddevice"

    sounddevice_module = types.SimpleNamespace(SoundDeviceLoopbackCapture=fake_sounddevice_capture)
    monkeypatch.setitem(sys.modules, "adsum.core.audio.sounddevice_backend", sounddevice_module)

    def failing_soundcard_capture(**_kwargs):
        raise CaptureError("soundcard unavailable in test")

    soundcard_module = types.SimpleNamespace(SoundCardLoopbackCapture=failing_soundcard_capture)
    monkeypatch.setitem(sys.modules, "adsum.core.audio.soundcard_backend", soundcard_module)

    request = CaptureRequest(
        channel="system",
        device='wasapi:"Speakers (Realtek(R) Audio)"?loopback=1',
        sample_rate=16_000,
        channels=1,
        backend="ffmpeg",
        chunk_seconds=0.1,
    )

    capture = create_capture(request)

    assert capture == "sounddevice"
    assert captured_kwargs["capture"]["target"] == "Speakers (Realtek(R) Audio)"
    assert captured_kwargs["capture"]["sample_rate"] == 48_000
    assert captured_kwargs["capture"]["channels"] == 2

