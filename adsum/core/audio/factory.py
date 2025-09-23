"""Factory helpers for constructing audio capture instances."""

from __future__ import annotations

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


def _create_sounddevice_capture(
    request: "CaptureRequest",
    *,
    info: Optional[CaptureInfo] = None,
    device_override: Optional[int | str | _Missing] = _MISSING,
) -> Optional[AudioCapture]:
    """Return a sounddevice capture instance for the provided request."""

    if device_override is _MISSING and _is_disabled_device(request.device):
        return None

    device = (
        _parse_device(request.device)
        if device_override is _MISSING
        else device_override
    )

    try:
        from .sounddevice_backend import SoundDeviceCapture
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise CaptureConfigurationError(
            "sounddevice dependency is required for audio capture"
        ) from exc

    capture_info = info or CaptureInfo(
        name=request.channel,
        sample_rate=request.sample_rate,
        channels=request.channels,
        device="default" if device is None else str(device),
    )
    capture_info.device = "default" if device is None else str(device)

    try:
        return SoundDeviceCapture(info=capture_info, device=device)
    except CaptureError as exc:
        raise CaptureConfigurationError(str(exc)) from exc


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
            LOGGER.info(
                "FFmpeg binary '%s' is unavailable; attempting sounddevice fallback for %s",
                settings.ffmpeg_binary,
                request.channel,
            )
            fallback = _create_sounddevice_capture(request, info=capture_info, device_override=None)
            if fallback is not None:
                return fallback
            raise CaptureConfigurationError(str(error)) from error

        try:
            return FFmpegCapture(
                info=capture_info,
                spec=spec,
                binary=resolved_binary,
                chunk_frames=chunk_frames,
            )
        except FFmpegBinaryNotFoundError as exc:
            LOGGER.info(
                "FFmpeg binary '%s' disappeared before start; falling back to sounddevice for %s",
                settings.ffmpeg_binary,
                request.channel,
            )
            try:
                fallback = _create_sounddevice_capture(request, info=capture_info, device_override=None)
            except CaptureConfigurationError as fallback_error:
                raise CaptureConfigurationError(
                    f"{exc}. Additionally, automatic fallback to sounddevice failed: {fallback_error}"
                ) from exc
            if fallback is not None:
                return fallback
            raise CaptureConfigurationError(str(exc)) from exc
        except CaptureError as exc:
            raise CaptureConfigurationError(str(exc)) from exc

    if backend == "sounddevice":
        capture = _create_sounddevice_capture(request)
        if capture is None:
            return None
        return capture

    raise CaptureConfigurationError(f"Unknown audio backend: {backend}")


__all__ = [
    "CaptureConfigurationError",
    "CaptureRequest",
    "create_capture",
    "DISABLED_DEVICE_SENTINEL",
]

