"""Tests for audio device utilities."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from adsum.core.audio import devices
from adsum import config
from adsum.core.audio.base import CaptureInfo


@dataclass
class _FakeStreamInfo:
    flags: int = 0


class _FakePointer:
    def __init__(self) -> None:
        self.contents = _FakeStreamInfo()


class _FakeSoundDeviceModule:
    def __init__(self) -> None:
        self.allow_loopback = True
        self._hostapis = [{"name": "Windows WASAPI"}]
        self._devices: list[dict] = []
        self._lib = type("Lib", (), {"paWinWasapiLoopback": 0x80})()
        self.created_settings: list[object] = []
        self.PortAudioError = RuntimeError

        module = self

        class _FakeWasapiSettings:
            def __init__(self, exclusive: bool = False, auto_convert: bool = False) -> None:
                module.created_settings.append(self)
                if module.allow_loopback:
                    self._streaminfo = _FakePointer()
                else:
                    self._streaminfo = None

        self.WasapiSettings = _FakeWasapiSettings

    def query_hostapis(self, index: int | None = None):  # pragma: no cover - trivial
        if index is None:
            return self._hostapis
        return self._hostapis[index]

    def query_devices(self):  # pragma: no cover - trivial
        return list(self._devices)


def test_format_device_table_fallback_contains_install_hint(monkeypatch) -> None:
    """When no devices are found the fallback message should include install hint."""

    monkeypatch.setattr(
        devices,
        "get_settings",
        lambda: config.Settings(audio_backend="sounddevice"),
    )
    monkeypatch.setattr(devices, "list_input_devices", lambda: [])

    message = devices.format_device_table()

    assert message.startswith("No input devices detected.")
    assert "pip install adsum[audio]" in message


def test_wasapi_loopback_fallback_without_keyword(monkeypatch) -> None:
    """Older sounddevice builds should enable WASAPI loopback via fallback logic."""

    fake_sd = _FakeSoundDeviceModule()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from adsum.core.audio.sounddevice_backend import (  # noqa: WPS433 - imported for test
        SoundDeviceCapture,
        wasapi_loopback_capable,
    )

    capture = SoundDeviceCapture(
        CaptureInfo(name="loopback", sample_rate=48_000, channels=2),
        device=0,
    )

    capture._configure_loopback(  # pylint: disable=protected-access
        {
            "hostapi": 0,
            "max_input_channels": 0,
            "max_output_channels": 2,
        }
    )

    assert capture._loopback_channels == 2  # pylint: disable=protected-access
    assert fake_sd.created_settings
    assert capture._extra_settings is fake_sd.created_settings[-1]
    assert (
        capture._extra_settings._streaminfo.contents.flags  # type: ignore[union-attr]
        == fake_sd._lib.paWinWasapiLoopback
    )
    assert wasapi_loopback_capable(fake_sd) is True

    fake_sd.allow_loopback = False
    fake_sd._devices = [  # pylint: disable=protected-access
        {
            "name": "Fake Output",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "hostapi": 0,
            "default_samplerate": 48_000.0,
        }
    ]

    assert devices.list_input_devices() == []
