"""Tests for audio capture factory helpers."""

from __future__ import annotations

import sys
import types

from adsum.core.audio.factory import CaptureRequest, create_capture


def test_sounddevice_capture_uses_default_device(monkeypatch) -> None:
    """Sounddevice backend should allow default device selection."""

    captured: dict[str, object] = {}

    class DummyCapture:
        def __init__(self, info, device) -> None:  # pragma: no cover - simple initializer
            captured["info"] = info
            captured["device"] = device

    fake_module = types.SimpleNamespace(SoundDeviceCapture=DummyCapture)
    monkeypatch.setitem(
        sys.modules,
        "adsum.core.audio.sounddevice_backend",
        fake_module,
    )

    request = CaptureRequest(
        channel="microphone",
        device=None,
        sample_rate=16_000,
        channels=1,
        backend="sounddevice",
    )

    capture = create_capture(request)

    assert capture is not None
    assert captured["device"] is None
    assert captured["info"].device == "default"


def test_sounddevice_capture_respects_disable_keywords(monkeypatch) -> None:
    """Disable keywords should prevent capture creation."""

    fake_module = types.SimpleNamespace(SoundDeviceCapture=object)
    monkeypatch.setitem(
        sys.modules,
        "adsum.core.audio.sounddevice_backend",
        fake_module,
    )

    request = CaptureRequest(
        channel="microphone",
        device="skip",
        sample_rate=16_000,
        channels=1,
        backend="sounddevice",
    )

    assert create_capture(request) is None


def test_ffmpeg_backend_falls_back_to_sounddevice(monkeypatch) -> None:
    """When FFmpeg is unavailable, the factory should switch to sounddevice."""

    import adsum.core.audio.factory as factory_module
    import adsum.core.audio.ffmpeg_backend as ffmpeg_backend

    class DummySettings:
        audio_backend = "ffmpeg"
        ffmpeg_binary = "ffmpeg"
        chunk_seconds = 0.5

    monkeypatch.setattr(factory_module, "get_settings", lambda: DummySettings())

    monkeypatch.setattr(ffmpeg_backend, "_resolve_binary", lambda binary: None)

    captured: dict[str, object] = {}

    class DummySoundDeviceCapture:
        def __init__(self, info, device) -> None:  # pragma: no cover - simple initializer
            captured["info"] = info
            captured["device"] = device

    fake_module = types.SimpleNamespace(SoundDeviceCapture=DummySoundDeviceCapture)
    monkeypatch.setitem(
        sys.modules,
        "adsum.core.audio.sounddevice_backend",
        fake_module,
    )

    request = CaptureRequest(
        channel="microphone",
        device="pulse:default",
        sample_rate=16_000,
        channels=1,
        backend="ffmpeg",
    )

    capture = create_capture(request)

    assert isinstance(capture, DummySoundDeviceCapture)
    assert captured["device"] is None
    assert captured["info"].device == "default"
