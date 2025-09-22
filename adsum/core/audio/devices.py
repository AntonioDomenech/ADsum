"""Helpers for enumerating audio devices using sounddevice when available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ...logging import get_logger

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
    for idx, info in enumerate(devices):
        if info["max_input_channels"] <= 0:
            continue
        hostapi = hostapis[info["hostapi"]]["name"] if hostapis else "unknown"
        name = info["name"]
        is_loopback = "loopback" in name.lower() or "monitor" in name.lower()
        results.append(
            DeviceInfo(
                id=idx,
                name=name,
                max_input_channels=info["max_input_channels"],
                default_samplerate=info.get("default_samplerate", 0.0),
                hostapi=hostapi,
                is_loopback=is_loopback,
            )
        )
    return results


def format_device_table() -> str:
    devices = list_input_devices()
    if not devices:
        return (
            "No input devices detected. Install optional audio support with "
            "`pip install adsum[audio]` and ensure audio hardware is accessible."
        )

    header = f"{'ID':>3} | {'Name':<40} | {'In':>2} | {'Rate':>7} | Host API | Loopback"
    lines = [header, "-" * len(header)]
    for device in devices:
        lines.append(
            f"{device.id:>3} | {device.name:<40.40} | {device.max_input_channels:>2} | "
            f"{int(device.default_samplerate):>7} | {device.hostapi:<8} | {('yes' if device.is_loopback else 'no'):>8}"
        )
    return "\n".join(lines)


__all__ = ["DeviceInfo", "list_input_devices", "format_device_table"]

