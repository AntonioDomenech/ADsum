"""Helpers for enumerating audio devices using sounddevice when available."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

from ...config import get_settings
from ...logging import get_logger
from .ffmpeg_backend import FFmpegBinaryNotFoundError, ensure_ffmpeg_available
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


@dataclass
class FFmpegDevice:
    """Structured information about an FFmpeg-reported capture source."""

    index: int
    name: str
    input_format: str
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    details: Optional[str] = None


def _quote_dshow_value(value: str) -> str:
    """Return a DirectShow-friendly quoted value."""

    trimmed = value.strip()
    if trimmed.startswith('"') and trimmed.endswith('"') and len(trimmed) >= 2:
        trimmed = trimmed[1:-1]
    escaped = trimmed.replace('"', '\\"')
    return f'"{escaped}"'


def recommended_ffmpeg_device_spec(device: FFmpegDevice) -> Optional[str]:
    """Return a suggested FFmpeg capture specification for the given device."""

    fmt = (device.input_format or "").strip().lower()
    if not fmt:
        return None

    if fmt == "dshow":
        alt = (device.details or "").strip()
        if alt:
            # Include the audio= prefix when omitted in alternative identifiers.
            if alt.lower().startswith("audio="):
                _, _, target_value = alt.partition("=")
            else:
                target_value = alt
            quoted = _quote_dshow_value(target_value)
            return f"dshow:audio={quoted}"
        if device.name:
            quoted = _quote_dshow_value(device.name)
            return f"dshow:audio={quoted}"
        return None

    if fmt == "avfoundation":
        if device.index is not None:
            return f"avfoundation:{device.index}"
        return None

    if fmt == "pulse":
        target = (device.details or device.name or "").strip()
        if not target:
            return None
        return f"pulse:{target}"

    if device.name:
        return f"{fmt}:{device.name}"
    return None


class FFmpegDeviceEnumerationError(RuntimeError):
    """Raised when FFmpeg cannot enumerate available devices."""


def _detect_ffmpeg_platform() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    return "unknown"


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
        try:
            ffmpeg_devices = list_ffmpeg_devices()
        except FFmpegBinaryNotFoundError as exc:
            LOGGER.warning("FFmpeg binary missing while listing devices: %s", exc)
            return format_ffmpeg_error_message(
                settings.ffmpeg_binary,
                f"Unable to launch FFmpeg for device enumeration: {exc}",
            )
        except FFmpegDeviceEnumerationError as exc:
            LOGGER.warning("FFmpeg device enumeration failed: %s", exc)
            return format_ffmpeg_error_message(
                settings.ffmpeg_binary,
                f"Unable to enumerate FFmpeg audio devices: {exc}",
            )

        if not ffmpeg_devices:
            return format_ffmpeg_error_message(
                settings.ffmpeg_binary,
                "No FFmpeg audio input devices were reported.",
            )

        return _format_ffmpeg_device_table(ffmpeg_devices)

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


def list_ffmpeg_devices() -> List[FFmpegDevice]:
    """Return audio capture devices reported by the configured FFmpeg binary."""

    settings = get_settings()
    binary = settings.ffmpeg_binary or "ffmpeg"
    executable = ensure_ffmpeg_available(binary) or binary

    platform = _detect_ffmpeg_platform()
    command: Sequence[str]
    parser: Callable[[str], List[FFmpegDevice]]
    if platform == "windows":
        command = [
            executable,
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        ]
        parser = _parse_ffmpeg_dshow_devices
    elif platform == "darwin":
        command = [
            executable,
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "avfoundation",
            "-i",
            "",
        ]
        parser = _parse_ffmpeg_avfoundation_devices
    elif platform == "linux":
        command = [
            executable,
            "-hide_banner",
            "-sources",
            "pulse",
        ]
        parser = _parse_ffmpeg_pulse_devices
    else:
        raise FFmpegDeviceEnumerationError(
            "FFmpeg device enumeration is not supported on this platform."
        )

    try:
        completed = subprocess.run(  # noqa: S603,S607 - trusted binary determined by settings
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - exercised via FFmpegBinaryNotFoundError
        raise FFmpegBinaryNotFoundError(binary) from exc
    except OSError as exc:
        raise FFmpegDeviceEnumerationError(str(exc)) from exc

    output = "\n".join(filter(None, [completed.stdout, completed.stderr]))
    devices = parser(output)
    return devices


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


def format_ffmpeg_error_message(binary: str, message: str) -> str:
    base = _format_ffmpeg_instructions(binary)
    return f"{base}\n\n{message}"


def _format_ffmpeg_device_table(devices: Sequence[FFmpegDevice]) -> str:
    header = f"{'ID':>3} | {'Name':<40} | {'In':>2} | {'Rate':>7} | Host API | Loopback"
    lines = [header, "-" * len(header)]
    for device in devices:
        display_name = device.name
        if device.details and device.details not in display_name:
            combined = f"{display_name} ({device.details})"
            display_name = combined
        channels = str(device.channels) if device.channels is not None else "--"
        sample_rate = (
            str(int(device.sample_rate))
            if device.sample_rate is not None and device.sample_rate > 0
            else "--"
        )
        lines.append(
            "{} | {} | {} | {} | {} | {}".format(
                f"{device.index:>3}",
                f"{display_name:<40.40}",
                f"{channels:>2}",
                f"{sample_rate:>7}",
                f"{device.input_format:<8.8}",
                f"{'n/a':>8}",
            )
        )
    return "\n".join(lines)


def _ffmpeg_payload(line: str) -> str:
    if "]" in line:
        _, payload = line.split("]", 1)
        return payload.strip()
    return line.strip()


def _parse_ffmpeg_dshow_devices(output: str) -> List[FFmpegDevice]:
    devices: List[FFmpegDevice] = []
    in_audio_section = False
    for line in output.splitlines():
        payload = _ffmpeg_payload(line)
        if not payload:
            continue
        lowered = payload.lower()
        if "directshow audio devices" in lowered:
            in_audio_section = True
            continue
        if "directshow video devices" in lowered:
            in_audio_section = False
            continue
        if not in_audio_section:
            continue
        alt_match = re.search(r"alternative name\s*:?-?\s*\"([^\"]+)\"", payload, re.IGNORECASE)
        if alt_match and devices:
            devices[-1].details = alt_match.group(1)
            continue
        name_match = re.search(r'"([^"]+)"', payload)
        if name_match:
            name = name_match.group(1)
            device = FFmpegDevice(
                index=len(devices),
                name=name,
                input_format="dshow",
            )
            devices.append(device)
    return devices


def _parse_ffmpeg_avfoundation_devices(output: str) -> List[FFmpegDevice]:
    devices: List[FFmpegDevice] = []
    in_audio_section = False
    for line in output.splitlines():
        payload = _ffmpeg_payload(line)
        if not payload:
            continue
        lowered = payload.lower()
        if "avfoundation audio devices" in lowered:
            in_audio_section = True
            continue
        if "avfoundation video devices" in lowered:
            in_audio_section = False
            continue
        if not in_audio_section:
            continue
        match = re.match(r"\[(\d+)\]\s*(.+)", payload)
        if not match:
            continue
        index = int(match.group(1))
        name = match.group(2).strip()
        devices.append(
            FFmpegDevice(
                index=index,
                name=name,
                input_format="avfoundation",
            )
        )
    devices.sort(key=lambda item: item.index)
    return devices


def _parse_ffmpeg_pulse_devices(output: str) -> List[FFmpegDevice]:
    devices: List[FFmpegDevice] = []
    in_sources = False
    current: Optional[FFmpegDevice] = None
    current_raw_name: Optional[str] = None

    def _finalize_current() -> None:
        nonlocal current, current_raw_name
        if current is None:
            return
        if current.name:
            if current_raw_name and current_raw_name != current.name:
                current.details = current_raw_name
        elif current_raw_name:
            current.name = current_raw_name
        devices.append(current)
        current = None
        current_raw_name = None

    for line in output.splitlines():
        payload = _ffmpeg_payload(line)
        if not payload:
            continue
        lowered = payload.lower()
        if lowered.startswith("sources"):
            in_sources = True
            _finalize_current()
            continue
        if lowered.startswith("sinks"):
            in_sources = False
            _finalize_current()
            break
        if not in_sources:
            continue

        source_match = re.match(r"source\s*#(\d+)", payload, re.IGNORECASE)
        index_match = re.match(r"(\d+):\s*(.+)", payload)
        if source_match:
            _finalize_current()
            current = FFmpegDevice(index=int(source_match.group(1)), name="", input_format="pulse")
            current_raw_name = None
            continue
        if index_match:
            _finalize_current()
            current = FFmpegDevice(
                index=int(index_match.group(1)),
                name=index_match.group(2).strip(),
                input_format="pulse",
            )
            current_raw_name = current.name or None
            continue
        if current is None:
            continue

        if lowered.startswith("name:"):
            value = payload.split(":", 1)[1].strip()
            current_raw_name = value or current_raw_name
            if not current.name:
                current.name = value
            continue
        if lowered.startswith("description:"):
            value = payload.split(":", 1)[1].strip()
            if value:
                current.name = value
            continue
        if lowered.startswith("sample spec:"):
            spec = payload.split(":", 1)[1].strip()
            channel_match = re.search(r"(\d+)ch", spec)
            if channel_match:
                current.channels = int(channel_match.group(1))
            rate_match = re.search(r"(\d+)hz", spec, re.IGNORECASE)
            if rate_match:
                try:
                    current.sample_rate = int(rate_match.group(1))
                except ValueError:
                    current.sample_rate = None
            continue
        if lowered.startswith("channels:") and current.channels is None:
            try:
                current.channels = int(payload.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                current.channels = None

    _finalize_current()
    devices.sort(key=lambda item: item.index)
    return devices


__all__ = [
    "DeviceInfo",
    "FFmpegDevice",
    "FFmpegDeviceEnumerationError",
    "list_input_devices",
    "list_ffmpeg_devices",
    "format_device_table",
    "format_ffmpeg_error_message",
    "recommended_ffmpeg_device_spec",
]

