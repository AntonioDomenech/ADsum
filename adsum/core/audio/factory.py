"""Factory helpers for constructing audio capture instances."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ...config import get_settings
from ...logging import get_logger
from .base import AudioCapture, CaptureError, CaptureInfo


LOGGER = get_logger(__name__)

DISABLE_DEVICE_KEYWORDS = {"skip", "none", "off", "disabled"}
DISABLED_DEVICE_SENTINEL = ":disabled:"


def _is_disabled_device(value: Optional[str]) -> bool:
    if value is None:
        return False
    if value == DISABLED_DEVICE_SENTINEL:
        return True
    return value.strip().lower() in DISABLE_DEVICE_KEYWORDS


class _Missing:
    pass


_MISSING = _Missing()
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

    if _is_disabled_device(request.device):
        return None

    if backend in {"", "none"}:
        return None

    if backend == "ffmpeg":
        if not request.device:
            raise CaptureConfigurationError("FFmpeg backend requires a device string")

        try:
            from .ffmpeg_backend import (
                FFmpegBinaryNotFoundError,
                FFmpegCapture,
                _resolve_binary,
                parse_ffmpeg_device,
            )
        except ImportError as exc:  # pragma: no cover - defensive
            raise CaptureConfigurationError("FFmpeg backend is unavailable") from exc

        info_device = (
            "default"
            if request.device is None or (isinstance(request.device, str) and not request.device.strip())
            else str(request.device)
        )

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

        resolved_binary = _resolve_binary(settings.ffmpeg_binary)

        if resolved_binary is None:
            error = FFmpegBinaryNotFoundError(settings.ffmpeg_binary)
            search_path = os.environ.get("PATH", "")
            LOGGER.warning(
                "FFmpeg binary '%s' is unavailable; install FFmpeg or set ADSUM_FFMPEG_BINARY "
                "to the executable path. Current PATH: %s",
                settings.ffmpeg_binary,
                search_path,
            )
            raise CaptureConfigurationError(str(error)) from error

        try:
            return FFmpegCapture(
                info=capture_info,
                spec=spec,
                binary=resolved_binary,
                chunk_frames=chunk_frames,
            )
        except FFmpegBinaryNotFoundError as exc:
            raise CaptureConfigurationError(str(exc)) from exc
        except CaptureError as exc:
            raise CaptureConfigurationError(str(exc)) from exc

    raise CaptureConfigurationError(
        f"Unknown audio backend: {backend}. Only 'ffmpeg' is supported."
    )


__all__ = [
    "CaptureConfigurationError",
    "CaptureRequest",
    "create_capture",
    "DISABLED_DEVICE_SENTINEL",
]

