"""Tests for audio device utilities."""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from types import SimpleNamespace

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


def test_format_device_table_accepts_custom_device_list(monkeypatch) -> None:
    """Providing an explicit device list should bypass automatic discovery."""

    monkeypatch.setattr(
        devices,
        "get_settings",
        lambda: config.Settings(audio_backend="sounddevice"),
    )

    custom_devices = [
        devices.DeviceInfo(
            id=7,
            name="Custom Microphone",
            max_input_channels=2,
            default_samplerate=48_000.0,
            hostapi="CoreAudio",
            is_loopback=False,
        )
    ]

    table = devices.format_device_table(custom_devices)

    assert "Custom Microphone" in table
    assert "  7 |" in table


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


def _mock_ffmpeg_run(output: str) -> SimpleNamespace:
    return SimpleNamespace(stdout="", stderr=output, returncode=1)


def test_list_ffmpeg_devices_parses_dshow_output(monkeypatch) -> None:
    """Windows FFmpeg listings should return structured DirectShow devices."""

    listing = textwrap.dedent(
        """
        [dshow @ 0x123] DirectShow video devices (some comment)
        [dshow @ 0x123]  "Integrated Camera"
        [dshow @ 0x123] DirectShow audio devices
        [dshow @ 0x123]  "Microphone (Realtek(R) Audio)" (audio)
        [dshow @ 0x123]     Alternative name "@device_cm_{123ABC}"
        [dshow @ 0x123]  "Stereo Mix (Realtek(R) Audio)" (audio)
        """
    ).strip()

    monkeypatch.setattr(devices, "ensure_ffmpeg_available", lambda binary: binary)
    monkeypatch.setattr(devices, "get_settings", lambda: config.Settings(ffmpeg_binary="ffmpeg"))
    monkeypatch.setattr(devices, "_detect_ffmpeg_platform", lambda: "windows")
    monkeypatch.setattr(devices.subprocess, "run", lambda *_, **__: _mock_ffmpeg_run(listing))

    results = devices.list_ffmpeg_devices()

    assert [device.name for device in results] == [
        "Microphone (Realtek(R) Audio)",
        "Stereo Mix (Realtek(R) Audio)",
    ]
    assert results[0].details == "@device_cm_{123ABC}"
    assert all(device.input_format == "dshow" for device in results)


def test_list_ffmpeg_devices_parses_avfoundation_output(monkeypatch) -> None:
    """macOS FFmpeg listings should surface AVFoundation audio devices."""

    listing = textwrap.dedent(
        """
        [AVFoundation input device @ 0x9] AVFoundation video devices:
        [AVFoundation input device @ 0x9] [0] FaceTime HD Camera
        [AVFoundation input device @ 0x9] AVFoundation audio devices:
        [AVFoundation input device @ 0x9] [1] Built-in Microphone
        [AVFoundation input device @ 0x9] [3] External USB Audio
        """
    ).strip()

    monkeypatch.setattr(devices, "ensure_ffmpeg_available", lambda binary: binary)
    monkeypatch.setattr(devices, "get_settings", lambda: config.Settings(ffmpeg_binary="ffmpeg"))
    monkeypatch.setattr(devices, "_detect_ffmpeg_platform", lambda: "darwin")
    monkeypatch.setattr(devices.subprocess, "run", lambda *_, **__: _mock_ffmpeg_run(listing))

    results = devices.list_ffmpeg_devices()

    assert [device.index for device in results] == [1, 3]
    assert [device.name for device in results] == [
        "Built-in Microphone",
        "External USB Audio",
    ]
    assert all(device.input_format == "avfoundation" for device in results)


def test_list_ffmpeg_devices_parses_pulse_output(monkeypatch) -> None:
    """Linux FFmpeg listings should parse PulseAudio sources with metadata."""

    listing = textwrap.dedent(
        """
        [pulse @ 0x7f] Sources:
        [pulse @ 0x7f] Source #0
        [pulse @ 0x7f]   Name: alsa_input.pci-0000_00_1b.0.analog-stereo
        [pulse @ 0x7f]   Description: Built-in Audio Analog Stereo
        [pulse @ 0x7f]   Sample Spec: s16le 2ch 44100Hz
        [pulse @ 0x7f] Source #2
        [pulse @ 0x7f]   Name: bluez_input.11_22_33_44_55_66.a2dp-source
        [pulse @ 0x7f]   Sample Spec: s16le 1ch 48000Hz
        [pulse @ 0x7f] Sinks:
        [pulse @ 0x7f] Sink #0
        """
    ).strip()

    monkeypatch.setattr(devices, "ensure_ffmpeg_available", lambda binary: binary)
    monkeypatch.setattr(devices, "get_settings", lambda: config.Settings(ffmpeg_binary="ffmpeg"))
    monkeypatch.setattr(devices, "_detect_ffmpeg_platform", lambda: "linux")
    monkeypatch.setattr(devices.subprocess, "run", lambda *_, **__: _mock_ffmpeg_run(listing))

    results = devices.list_ffmpeg_devices()

    assert [device.index for device in results] == [0, 2]
    assert results[0].name == "Built-in Audio Analog Stereo"
    assert results[0].details == "alsa_input.pci-0000_00_1b.0.analog-stereo"
    assert results[0].channels == 2
    assert results[0].sample_rate == 44100
    assert results[1].name == "bluez_input.11_22_33_44_55_66.a2dp-source"
    assert results[1].channels == 1
    assert results[1].sample_rate == 48000


def test_format_device_table_ffmpeg_formats_results(monkeypatch) -> None:
    """format_device_table should render FFmpeg listings in a tabular layout."""

    monkeypatch.setattr(
        devices,
        "get_settings",
        lambda: config.Settings(audio_backend="ffmpeg", ffmpeg_binary="ffmpeg"),
    )
    monkeypatch.setattr(
        devices,
        "list_ffmpeg_devices",
        lambda: [
            devices.FFmpegDevice(
                index=0,
                name="Built-in Microphone",
                input_format="avfoundation",
            ),
            devices.FFmpegDevice(
                index=1,
                name="USB Interface",
                input_format="pulse",
                channels=2,
                sample_rate=48_000,
                details="alsa_input.usb-123",
            ),
        ],
    )

    table = devices.format_device_table()

    assert "Built-in Microphone" in table
    assert "USB Interface" in table
    assert "48000" in table
    assert "pulse" in table


def test_format_device_table_ffmpeg_handles_errors(monkeypatch) -> None:
    """Failures during FFmpeg enumeration should include diagnostic output."""

    monkeypatch.setattr(
        devices,
        "get_settings",
        lambda: config.Settings(audio_backend="ffmpeg", ffmpeg_binary="ffmpeg"),
    )
    monkeypatch.setattr(
        devices,
        "list_ffmpeg_devices",
        lambda: (_ for _ in ()).throw(devices.FFmpegDeviceEnumerationError("boom")),
    )

    message = devices.format_device_table()

    assert "FFmpeg backend is active" in message
    assert "Unable to enumerate FFmpeg audio devices" in message
    assert "boom" in message


def test_format_device_table_ffmpeg_empty_listing(monkeypatch) -> None:
    """Empty FFmpeg results should surface an explicit notice."""

    monkeypatch.setattr(
        devices,
        "get_settings",
        lambda: config.Settings(audio_backend="ffmpeg", ffmpeg_binary="ffmpeg"),
    )
    monkeypatch.setattr(devices, "list_ffmpeg_devices", lambda: [])

    message = devices.format_device_table()

    assert "No FFmpeg audio input devices were reported." in message


def test_recommended_ffmpeg_device_spec_prefers_alternative_name() -> None:
    """Alternative DirectShow identifiers should be used when available."""

    device = devices.FFmpegDevice(
        index=1,
        name="Microphone (Realtek(R) Audio)",
        input_format="dshow",
        details="@device_cm_{123ABC}",
    )

    spec = devices.recommended_ffmpeg_device_spec(device)

    assert spec == 'dshow:audio="@device_cm_{123ABC}"'


def test_recommended_ffmpeg_device_spec_quotes_primary_name() -> None:
    """DirectShow devices without alternative names should quote the display name."""

    device = devices.FFmpegDevice(
        index=0,
        name="Stereo Mix (Realtek(R) Audio)",
        input_format="dshow",
    )

    spec = devices.recommended_ffmpeg_device_spec(device)

    assert spec == 'dshow:audio="Stereo Mix (Realtek(R) Audio)"'


def test_recommended_ffmpeg_device_spec_handles_other_backends() -> None:
    """macOS and PulseAudio devices should produce usable specifications."""

    avfoundation = devices.FFmpegDevice(index=3, name="USB Audio", input_format="avfoundation")
    pulse = devices.FFmpegDevice(
        index=1,
        name="Built-in Audio Analog Stereo",
        input_format="pulse",
        details="alsa_input.pci-0000_00_1b.0.analog-stereo",
    )

    assert devices.recommended_ffmpeg_device_spec(avfoundation) == "avfoundation:3"
    assert (
        devices.recommended_ffmpeg_device_spec(pulse)
        == "pulse:alsa_input.pci-0000_00_1b.0.analog-stereo"
    )
