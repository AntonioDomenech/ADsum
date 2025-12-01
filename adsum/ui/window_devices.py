"""Helpers for device selection and probing used by the window UI."""

from __future__ import annotations

import contextlib
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

from ..config import Settings
from ..core.audio.base import CaptureError
from ..core.audio.devices import (
    DeviceInfo,
    FFmpegDevice,
    FFmpegDeviceEnumerationError,
    format_device_table,
    format_ffmpeg_error_message,
    list_ffmpeg_devices,
    list_input_devices,
    recommended_ffmpeg_device_spec,
)
from ..core.audio.factory import (
    CaptureConfigurationError,
    CaptureRequest,
    DISABLED_DEVICE_SENTINEL,
    create_capture,
)
from ..core.audio.ffmpeg_backend import (
    FFmpegBinaryNotFoundError,
    parse_ffmpeg_device,
)
from ..logging import get_logger
from .shared import normalize_device_value

LOGGER = get_logger(__name__)


def render_device_table(settings: Settings) -> str:
    """Return a human-readable device table, handling FFmpeg errors."""

    try:
        return format_device_table()
    except FFmpegBinaryNotFoundError as exc:
        LOGGER.error("FFmpeg binary unavailable while listing devices: %s", exc)
        message = f"Unable to launch FFmpeg for device enumeration: {exc}"
        return format_ffmpeg_error_message(settings.ffmpeg_binary, message)
    except FFmpegDeviceEnumerationError as exc:
        LOGGER.error("FFmpeg device enumeration failed: %s", exc)
        message = f"Unable to enumerate FFmpeg audio devices: {exc}"
        return format_ffmpeg_error_message(settings.ffmpeg_binary, message)


def load_ffmpeg_devices_for_options(settings: Settings) -> List[FFmpegDevice]:
    """Return FFmpeg-reported devices, suppressing errors to keep UI responsive."""

    try:
        return list_ffmpeg_devices()
    except FFmpegBinaryNotFoundError as exc:
        LOGGER.warning("FFmpeg binary unavailable while building device options: %s", exc)
    except FFmpegDeviceEnumerationError as exc:
        LOGGER.warning("FFmpeg device enumeration failed while building device options: %s", exc)
    return []


def probe_device_capture(
    device: str,
    *,
    sample_rate: int,
    channels: int,
    backend: Optional[str],
    chunk_seconds: float,
) -> Tuple[bool, str]:
    """Attempt to capture a short chunk from ``device``."""

    request = CaptureRequest(
        channel="probe",
        device=device,
        sample_rate=sample_rate,
        channels=channels,
        backend=backend,
        chunk_seconds=chunk_seconds,
    )

    try:
        capture = create_capture(request)
    except CaptureConfigurationError as exc:
        return False, f"configuration error: {exc}"

    if capture is None:
        return False, "no capture backend available"

    try:
        capture.start()
        chunk = None
        deadline = time.time() + 1.5
        while time.time() < deadline:
            try:
                chunk = capture.read(timeout=0.3)
            except CaptureError as exc:
                return False, f"read failed: {exc}"
            if chunk is not None and getattr(chunk, "size", 0) > 0:
                return True, ""
        return False, "no audio data received"
    except CaptureError as exc:
        return False, f"capture error: {exc}"
    except Exception as exc:  # pragma: no cover - depends on runtime backend
        LOGGER.exception("Unexpected error while probing device %s: %s", device, exc)
        return False, str(exc)
    finally:
        with contextlib.suppress(Exception):
            capture.stop()
        with contextlib.suppress(Exception):
            capture.close()


def auto_detect_working_devices(
    settings: Settings,
    *,
    sample_rate: int,
    channels: int,
    chunk_seconds: float,
    info_callback: Callable[[str], None],
) -> Tuple[List[DeviceInfo], str]:
    """Probe devices, returning working inputs and a report string."""

    backend = (settings.audio_backend or "").strip().lower()

    if backend == "ffmpeg":
        message = render_device_table(settings)
        info_callback("FFmpeg audio backend active; skipping automatic device probing.")
        return [], message

    devices = list_input_devices()

    if not devices:
        message = (
            "No audio input devices were detected by the legacy probe. Provide "
            "an FFmpeg capture specification manually when prompted."
        )
        info_callback("No audio input devices were detected before starting the session.")
        return [], message

    info_callback("Testing detected audio devices before starting the recording wizard...")

    working: List[DeviceInfo] = []
    failed: List[Tuple[DeviceInfo, str]] = []

    for device in devices:
        device_spec = str(device.id)
        label = f"[{device.id}] {device.name}"
        info_callback(f"Probing {label}...")
        success, reason = probe_device_capture(
            device_spec,
            sample_rate=sample_rate,
            channels=channels,
            backend=settings.audio_backend,
            chunk_seconds=chunk_seconds,
        )
        if success:
            working.append(device)
            info_callback(f"{label} is ready for use.")
        else:
            failed.append((device, reason))
            reason_text = reason or "unavailable"
            info_callback(f"{label} will be skipped: {reason_text}.")

    if not working:
        info_callback(
            "No working audio inputs were detected. You can proceed with system defaults or "
            "disable individual channels."
        )

    report = format_device_probe_report(working, failed)
    return working, report


def format_device_probe_report(
    working: Iterable[DeviceInfo],
    failed: Iterable[Tuple[DeviceInfo, str]],
) -> str:
    """Return a textual summary of probe results."""

    working_list = list(working)
    failed_list = list(failed)

    sections: List[str] = []

    if working_list:
        sections.append("Working audio input devices:")
        sections.append(format_device_table(working_list))
    else:
        sections.append("No working audio input devices were detected.")

    if failed_list:
        lines = ["", "Devices skipped after testing:"]
        for device, reason in failed_list:
            detail = reason or "unavailable"
            lines.append(f"  [{device.id}] {device.name} — {detail}")
        sections.extend(lines)

    return "\n".join(sections)


def build_device_option_map(
    devices: Optional[Iterable[DeviceInfo]],
    ffmpeg_devices: Optional[Iterable[FFmpegDevice]] = None,
) -> Dict[str, Optional[str]]:
    """Combine PortAudio and FFmpeg devices into a selector-friendly map."""

    options: Dict[str, Optional[str]] = {"Use system default": None}

    if os.name == "nt":
        options["Default system output (WASAPI loopback)"] = "wasapi:default?loopback=1"

    options["Disable capture"] = DISABLED_DEVICE_SENTINEL

    for device in devices or []:
        label = f"[{device.id}] {device.name}"
        if device.is_loopback:
            label += " (loopback)"
        value = str(device.id)
        normalized = normalize_device_value(value)
        options[label] = normalized if normalized is not None else value

    for device in ffmpeg_devices or []:
        spec = recommended_ffmpeg_device_spec(device)
        if not spec:
            continue
        label = format_ffmpeg_option_label(device)
        normalized = normalize_device_value(spec)
        options[label] = normalized if normalized is not None else spec

    return options


def format_ffmpeg_option_label(device: FFmpegDevice) -> str:
    """Return a descriptive label for an FFmpeg device."""

    base_name = device.name or device.details or "Unnamed device"
    label = f"[{device.index}] {base_name}"

    descriptors: List[str] = []
    if device.input_format:
        descriptors.append(device.input_format)
    if getattr(device, "loopback", False):
        descriptors.append("loopback")
    if device.channels:
        descriptors.append(f"{device.channels}ch")
    if device.sample_rate:
        descriptors.append(f"{device.sample_rate} Hz")
    if device.details and device.details not in base_name:
        descriptors.append(device.details)

    if descriptors:
        label = f"{label} — {', '.join(descriptors)}"

    return label


def format_ffmpeg_preview(
    mic: Optional[str],
    system: Optional[str],
    *,
    sample_rate: int,
    channels: int,
) -> str:
    """Return a preview of the FFmpeg specs used for mic/system channels."""

    sections = [
        describe_ffmpeg_channel("Microphone input", mic, sample_rate=sample_rate, channels=channels),
        describe_ffmpeg_channel("System audio", system, sample_rate=sample_rate, channels=channels),
    ]
    return "\n\n".join(section for section in sections if section).strip()


def describe_ffmpeg_channel(
    label: str,
    device: Optional[str],
    *,
    sample_rate: int,
    channels: int,
) -> str:
    """Return a verbose breakdown of an FFmpeg capture specification."""

    normalized = normalize_device_value(device)
    lines = [f"{label}:"]

    if normalized is None:
        lines.append("  Using system default input.")
        return "\n".join(lines)

    if normalized == DISABLED_DEVICE_SENTINEL:
        lines.append("  Capture disabled for this channel.")
        return "\n".join(lines)

    original_value = normalized
    lines.append(f"  Entered: {original_value}")

    try:
        spec = parse_ffmpeg_device(
            original_value,
            default_sample_rate=sample_rate,
            default_channels=channels,
        )
    except CaptureError as exc:
        lines.append(f"  Error: {exc}")
        return "\n".join(lines)

    target = f"{spec.input_format}:{spec.input_target}"
    original_split = urlsplit(original_value)
    if not original_split.scheme:
        lines.append(f"  Inferred target: {target}")
    else:
        lines.append(f"  Target: {target}")

    query_parts = [
        f"sample_rate={spec.sample_rate}",
        f"channels={spec.channels}",
        f"sample_fmt={spec.sample_format}",
    ]
    if spec.chunk_frames is not None:
        query_parts.append(f"chunk_frames={spec.chunk_frames}")

    normalized_spec = target
    if query_parts:
        normalized_spec = f"{target}?{'&'.join(query_parts)}"
    lines.append(f"  Normalised spec: {normalized_spec}")
    lines.append(
        f"  Stream parameters: {spec.channels} channel(s) @ {spec.sample_rate} Hz ({spec.sample_format})"
    )

    if spec.args_before_input:
        lines.append("  Extra input args: " + " ".join(spec.args_before_input))
    if spec.args_after_input:
        lines.append("  Extra output args: " + " ".join(spec.args_after_input))

    return "\n".join(lines)


__all__ = [
    "auto_detect_working_devices",
    "build_device_option_map",
    "describe_ffmpeg_channel",
    "format_device_probe_report",
    "format_ffmpeg_option_label",
    "format_ffmpeg_preview",
    "load_ffmpeg_devices_for_options",
    "probe_device_capture",
    "render_device_table",
]
