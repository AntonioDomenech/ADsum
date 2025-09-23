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
