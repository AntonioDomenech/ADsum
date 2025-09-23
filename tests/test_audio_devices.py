from types import SimpleNamespace
import textwrap

import pytest

from types import SimpleNamespace
import textwrap

import pytest

from adsum import config
from adsum.core.audio import devices


def _mock_ffmpeg_run(output: str) -> SimpleNamespace:
    return SimpleNamespace(stdout="", stderr=output, returncode=1)


def test_format_device_table_uses_ffmpeg_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device tables should be populated from FFmpeg listings."""

    fake_devices = [
        devices.FFmpegDevice(index=0, name="USB Mic", input_format="dshow", channels=1),
        devices.FFmpegDevice(index=1, name="Stereo Mix", input_format="dshow", channels=2),
    ]

    monkeypatch.setattr(devices, "get_settings", lambda: config.Settings(ffmpeg_binary="ffmpeg"))
    monkeypatch.setattr(devices, "list_ffmpeg_devices", lambda: fake_devices)

    table = devices.format_device_table()

    assert "USB Mic" in table
    assert "Stereo Mix" in table


def test_format_device_table_warns_for_unsupported_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backends other than FFmpeg should report a helpful message."""

    monkeypatch.setattr(devices, "get_settings", lambda: config.Settings(audio_backend="portaudio"))

    message = devices.format_device_table()

    assert "Audio backend 'portaudio'" in message
    assert "ADSUM_AUDIO_BACKEND=ffmpeg" in message


def test_list_ffmpeg_devices_parses_dshow_output(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_list_ffmpeg_devices_handles_dshow_without_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Newer FFmpeg builds omit DirectShow headers but still list devices."""

    listing = textwrap.dedent(
        """
        [dshow @ 0x123] "Integrated Webcam" (video)
        [dshow @ 0x123]   Alternative name "@device_pnp_WEBCAM"
        [dshow @ 0x123] "Mezcla estéreo (2- Realtek(R) Audio)" (audio)
        [dshow @ 0x123]   Alternative name "@device_cm_{AUDIO_MIX}"
        [dshow @ 0x123] "Varios micrófonos (2- Realtek(R) Audio)" (audio)
        [dshow @ 0x123]   Alternative name "@device_cm_{AUDIO_MIC}"
        Error opening input file dummy.
        """
    ).strip()

    monkeypatch.setattr(devices, "ensure_ffmpeg_available", lambda binary: binary)
    monkeypatch.setattr(devices, "get_settings", lambda: config.Settings(ffmpeg_binary="ffmpeg"))
    monkeypatch.setattr(devices, "_detect_ffmpeg_platform", lambda: "windows")
    monkeypatch.setattr(devices.subprocess, "run", lambda *_, **__: _mock_ffmpeg_run(listing))

    results = devices.list_ffmpeg_devices()

    assert [device.name for device in results] == [
        "Mezcla estéreo (2- Realtek(R) Audio)",
        "Varios micrófonos (2- Realtek(R) Audio)",
    ]
    assert [device.details for device in results] == ["@device_cm_{AUDIO_MIX}", "@device_cm_{AUDIO_MIC}"]


def test_list_ffmpeg_devices_parses_avfoundation_output(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_list_ffmpeg_devices_parses_pulse_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux FFmpeg listings should expose PulseAudio sources."""

    listing = textwrap.dedent(
        """
        [pulse] Sources:
        [pulse]  0: Monitor of Built-in Audio Analog Stereo
        [pulse]     Flags: DECIBEL_RANGE VOLUME MUTE
        [pulse]  1: USB Microphone
        [pulse]     Sample spec: s16le 2ch 48000Hz
        [pulse] Sinks:
        """
    ).strip()

    monkeypatch.setattr(devices, "ensure_ffmpeg_available", lambda binary: binary)
    monkeypatch.setattr(devices, "get_settings", lambda: config.Settings(ffmpeg_binary="ffmpeg"))
    monkeypatch.setattr(devices, "_detect_ffmpeg_platform", lambda: "linux")
    monkeypatch.setattr(devices.subprocess, "run", lambda *_, **__: _mock_ffmpeg_run(listing))

    results = devices.list_ffmpeg_devices()

    assert [device.name for device in results] == [
        "Monitor of Built-in Audio Analog Stereo",
        "USB Microphone",
    ]
    assert results[0].input_format == "pulse"
    assert results[1].channels == 2

