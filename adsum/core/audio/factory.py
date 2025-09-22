"""Factory helpers for constructing audio capture instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import AudioCapture, CaptureError, CaptureInfo


class CaptureConfigurationError(RuntimeError):
    """Raised when a capture stream cannot be configured."""


@dataclass
class CaptureRequest:
    """Description of a capture channel requested by the user."""

    channel: str
    device: Optional[str]
    sample_rate: int
    channels: int


def _parse_device(device: Optional[str]) -> Optional[int | str]:
    if device is None:
        return None
    device = device.strip()
    if not device:
        return None
    if device.isdigit():
        return int(device)
    return device


def create_capture(request: CaptureRequest) -> Optional[AudioCapture]:
    """Create an :class:`AudioCapture` implementation for the given request."""

    device = _parse_device(request.device)
    if device is None:
        return None

    try:
        from .sounddevice_backend import SoundDeviceCapture
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise CaptureConfigurationError(
            "sounddevice dependency is required for audio capture"
        ) from exc

    capture_info = CaptureInfo(
        name=request.channel,
        sample_rate=request.sample_rate,
        channels=request.channels,
        device=str(request.device),
    )

    try:
        return SoundDeviceCapture(info=capture_info, device=device)
    except CaptureError as exc:
        raise CaptureConfigurationError(str(exc)) from exc


__all__ = ["CaptureConfigurationError", "CaptureRequest", "create_capture"]

