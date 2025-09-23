"""Factory helpers for constructing audio capture instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...config import get_settings
from .base import AudioCapture, CaptureError, CaptureInfo


DISABLE_DEVICE_KEYWORDS = {"skip", "none", "off", "disabled"}
DISABLED_DEVICE_SENTINEL = ":disabled:"


def _is_disabled_device(value: Optional[str]) -> bool:
    if value is None:
        return False
    if value == DISABLED_DEVICE_SENTINEL:
        return True
    return value.strip().lower() in DISABLE_DEVICE_KEYWORDS


class CaptureConfigurationError(RuntimeError):
    """Raised when a capture stream cannot be configured."""


@dataclass
class CaptureRequest:
    """Description of a capture channel requested by the user."""

    channel: str
    device: Optional[str]
    sample_rate: int
    channels: int
    backend: Optional[str] = None
    chunk_seconds: Optional[float] = None


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

    settings = get_settings()
    backend = (request.backend or settings.audio_backend or "").strip().lower()

    if backend in {"", "none"}:
        return None

    if backend == "ffmpeg":
        if not request.device:
            raise CaptureConfigurationError("FFmpeg backend requires a device string")

        try:
            from .ffmpeg_backend import FFmpegCapture, parse_ffmpeg_device
        except ImportError as exc:  # pragma: no cover - defensive
            raise CaptureConfigurationError("FFmpeg backend is unavailable") from exc

        info_device: Optional[str]
        if request.device is None or (
            isinstance(request.device, str) and not request.device.strip()
        ):
            info_device = "default"
        else:
            info_device = str(request.device)

        capture_info = CaptureInfo(
            name=request.channel,
            sample_rate=request.sample_rate,
            channels=request.channels,
            device=info_device,
        )

        try:
            spec = parse_ffmpeg_device(
                request.device,
                default_sample_rate=capture_info.sample_rate,
                default_channels=capture_info.channels,
            )
        except CaptureError as exc:
            raise CaptureConfigurationError(str(exc)) from exc

        chunk_seconds = request.chunk_seconds
        if chunk_seconds is None:
            chunk_seconds = settings.chunk_seconds

        chunk_frames = max(int(spec.sample_rate * max(chunk_seconds, 0.001)), 1)
        if spec.chunk_frames is not None:
            chunk_frames = max(spec.chunk_frames, 1)

        try:
            return FFmpegCapture(
                info=capture_info,
                spec=spec,
                binary=settings.ffmpeg_binary,
                chunk_frames=chunk_frames,
            )
        except CaptureError as exc:
            raise CaptureConfigurationError(str(exc)) from exc

    if backend == "sounddevice":
        if _is_disabled_device(request.device):
            return None

        device = _parse_device(request.device)

        try:
            from .sounddevice_backend import SoundDeviceCapture
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise CaptureConfigurationError(
                "sounddevice dependency is required for audio capture"
            ) from exc

        info_device = "default" if device is None else str(device)

        capture_info = CaptureInfo(
            name=request.channel,
            sample_rate=request.sample_rate,
            channels=request.channels,
            device=info_device,
        )

        try:
            return SoundDeviceCapture(info=capture_info, device=device)
        except CaptureError as exc:
            raise CaptureConfigurationError(str(exc)) from exc

    raise CaptureConfigurationError(f"Unknown audio backend: {backend}")


__all__ = [
    "CaptureConfigurationError",
    "CaptureRequest",
    "create_capture",
    "DISABLED_DEVICE_SENTINEL",
]

