"""Helpers for enumerating audio devices using sounddevice when available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from ...config import get_settings
from ...logging import get_logger
from .sounddevice_backend import wasapi_loopback_capable

LOGGER = get_logger(__name__)


@dataclass
class DeviceInfo:
    id: int
    name: str
    max_input_channels: int
    default_samplerate: float
    hostapi: str
    is_loopback: bool


def list_input_devices() -> List[DeviceInfo]:
    try:
        import sounddevice as sd
    except ImportError:
        LOGGER.warning("sounddevice not installed; cannot list devices")
        return []

    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    results: List[DeviceInfo] = []
    wasapi_loopback_supported: Optional[bool] = None

    for idx, info in enumerate(devices):
        hostapi = hostapis[info["hostapi"]]["name"] if hostapis else "unknown"
        name = info["name"]
        max_input = int(info.get("max_input_channels") or 0)
        max_output = int(info.get("max_output_channels") or 0)
        is_loopback = "loopback" in name.lower() or "monitor" in name.lower()

        if max_input <= 0:
            if "wasapi" in hostapi.lower() and max_output > 0:
                if wasapi_loopback_supported is None:
                    wasapi_loopback_supported = wasapi_loopback_capable(sd)
                if wasapi_loopback_supported:
                    # Windows WASAPI output devices support loopback capture
                    # when explicitly requested. Surface them so users can
                    # select the system playback stream.
                    max_input = max_output
                    is_loopback = True
                else:
                    LOGGER.debug(
                        "Skipping WASAPI output %s; loopback unsupported by sounddevice",
                        name,
                    )
                    continue
            else:
                continue
        results.append(
            DeviceInfo(
                id=idx,
                name=name,
                max_input_channels=max_input,
                default_samplerate=info.get("default_samplerate", 0.0),
                hostapi=hostapi,
                is_loopback=is_loopback,
            )
        )
    return results


def format_device_table(devices: Optional[Iterable[DeviceInfo]] = None) -> str:
    settings = get_settings()
    backend = (settings.audio_backend or "").lower()

    if backend == "ffmpeg" and devices is None:
        return _format_ffmpeg_instructions(settings.ffmpeg_binary)

    if devices is None:
        device_list = list_input_devices()
    else:
        device_list = list(devices)

    if not device_list:
        return (
            "No input devices detected. Install optional audio support with "
            "`pip install adsum[audio]` and ensure audio hardware is accessible."
        )

    header = f"{'ID':>3} | {'Name':<40} | {'In':>2} | {'Rate':>7} | Host API | Loopback"
    lines = [header, "-" * len(header)]
    for device in device_list:
        lines.append(
            f"{device.id:>3} | {device.name:<40.40} | {device.max_input_channels:>2} | "
            f"{int(device.default_samplerate):>7} | {device.hostapi:<8} | {('yes' if device.is_loopback else 'no'):>8}"
        )
    return "\n".join(lines)


def _format_ffmpeg_instructions(binary: str) -> str:
    message = [
        "FFmpeg backend is active. Provide a capture specification for each channel.",
        "The format follows: <input-format>:<input-target>?option=value&...",
        "Examples:",
        "  pulse:bluez_source.XX?sample_rate=48000&channels=2",
        "  dshow:audio=Bluetooth Headset?sample_rate=48000&channels=1",
        "  avfoundation:0?channels=1",
        "If FFmpeg is not installed, download a build from https://ffmpeg.org/download.html",
        "and add the 'bin' directory to PATH or set ADSUM_FFMPEG_BINARY to the full path.",
        "Additional FFmpeg arguments can be provided with args= or opt_/flag_ parameters.",
        "Set ADSUM_AUDIO_BACKEND=sounddevice if you prefer the legacy PortAudio backend.",
        f"Using FFmpeg binary: {binary}",
        "",
        "Discover devices with:",
        f"  Windows: {binary} -hide_banner -list_devices true -f dshow -i dummy",
        f"  macOS:   {binary} -hide_banner -list_devices true -f avfoundation -i \"\"",
        f"  Linux:   {binary} -hide_banner -sources pulse",
    ]
    return "\n".join(message)


__all__ = ["DeviceInfo", "list_input_devices", "format_device_table"]

