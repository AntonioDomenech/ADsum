"""Shared FFmpeg helpers used across capture and UI layers."""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import urlopen

from .base import CaptureError
from ...config import get_settings
from ...logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FFmpegDeviceSpec:
    """Parsed representation of an FFmpeg capture target."""

    input_format: str
    input_target: str
    args_before_input: List[str]
    input_options: List[str]
    args_after_input: List[str]
    sample_rate: int
    channels: int
    sample_format: str
    chunk_frames: Optional[int]


class FFmpegBinaryNotFoundError(CaptureError):
    """Raised when the FFmpeg executable cannot be located."""

    def __init__(self, requested: str) -> None:
        message = (
            "FFmpeg binary "
            f"'{requested or 'ffmpeg'}' was not found. Install FFmpeg and ensure the "
            "executable is on PATH, or set ADSUM_FFMPEG_BINARY to the full path to the "
            "ffmpeg executable."
        )
        super().__init__(message)
        self.requested = requested


def _detect_platform() -> str:
    """Return a simplified platform identifier used for heuristics."""

    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    return "unknown"


def _lookup_ffmpeg_device_name(index: int) -> Optional[str]:
    """Return the FFmpeg reported name for the given index when available."""

    try:
        from .devices import list_ffmpeg_devices
    except Exception:  # pragma: no cover - defensive import guard
        return None

    try:
        devices = list_ffmpeg_devices()
    except Exception:  # pragma: no cover - runtime enumeration errors
        return None

    for device in devices:
        if device.index == index:
            return device.name
    return None


def _quote_windows_device_name(name: str) -> str:
    """Return a quoted DirectShow device name with escaped quotes."""

    trimmed = name.strip()
    if trimmed.startswith('"') and trimmed.endswith('"') and len(trimmed) >= 2:
        trimmed = trimmed[1:-1]
    escaped = trimmed.replace('"', '\\"')
    return f'"{escaped}"'


def _guess_ffmpeg_device_target(device: str) -> Optional[str]:
    """Return a best-effort FFmpeg specification for targets lacking a scheme."""

    platform = _detect_platform()
    base = device.strip()
    if not base:
        return None

    if platform == "windows":
        target = base
        if base.lower().startswith("audio="):
            name = base[6:]
            target = f"audio={_quote_windows_device_name(name)}"
        else:
            resolved = base
            if base.isdigit():
                lookup = _lookup_ffmpeg_device_name(int(base))
                if lookup:
                    resolved = lookup
            target = f"audio={_quote_windows_device_name(resolved)}"
        return f"dshow:{target}"

    if platform == "darwin":
        if base.isdigit():
            return f"avfoundation:{base}"
        return None

    if platform == "linux":
        return f"pulse:{base}"

    return None


def _guess_ffmpeg_device_spec(device: str) -> Optional[str]:
    """Return a device specification with scheme if one can be inferred."""

    base, sep, query = device.partition("?")
    target = _guess_ffmpeg_device_target(base)
    if not target:
        return None
    if sep:
        return f"{target}?{query}"
    return target


def _strip_wrapping_quotes(value: str) -> str:
    """Return ``value`` without a single leading/trailing quote pair."""

    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
        trimmed = trimmed[1:-1]
    # Handle common escaped quote representations produced by shells or config files.
    trimmed = trimmed.replace(r'\"', '"').replace(r"\'", "'")
    return trimmed


def _normalise_input_target(input_format: str, target: str) -> str:
    """Return the FFmpeg input target with any wrapping quotes removed."""

    trimmed = target.strip()
    if not trimmed:
        return trimmed

    if input_format.lower() == "dshow":
        prefix, sep, remainder = trimmed.partition("=")
        if sep:
            remainder = _strip_wrapping_quotes(remainder)
            return f"{prefix}{sep}{remainder.strip()}"
        return _strip_wrapping_quotes(trimmed)

    return _strip_wrapping_quotes(trimmed)


def parse_ffmpeg_device(
    device: str,
    *,
    default_sample_rate: int,
    default_channels: int,
) -> FFmpegDeviceSpec:
    """Return a :class:`FFmpegDeviceSpec` parsed from the user supplied string."""

    if not device:
        raise CaptureError("FFmpeg backend requires a device specification")

    normalized_device = device.strip()
    split = urlsplit(normalized_device)

    if not split.scheme:
        guessed = _guess_ffmpeg_device_spec(normalized_device)
        if guessed:
            LOGGER.debug(
                "Normalised FFmpeg device specification '%s' -> '%s'", device, guessed
            )
            normalized_device = guessed
            split = urlsplit(normalized_device)

    if not split.scheme:
        raise CaptureError(
            "FFmpeg device specification must start with an input format, "
            "for example 'pulse:bluez_source.XX' or 'dshow:audio=Device'",
        )

    input_format = split.scheme
    input_target = _normalise_input_target(input_format, (split.netloc + split.path).strip())

    if not input_target:
        raise CaptureError("FFmpeg device specification must include a device identifier")

    args_before: List[str] = []
    args_after: List[str] = []
    input_options: List[str] = []
    sample_rate = int(default_sample_rate)
    channels = int(default_channels)
    sample_format = "f32le"
    chunk_frames: Optional[int] = None
    pending_chunk_ms: Optional[float] = None
    sample_rate_overridden = False
    channels_overridden = False
    loopback_requested = False

    for key, value in parse_qsl(split.query, keep_blank_values=True):
        if key == "sample_rate" and value:
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise CaptureError(f"Invalid FFmpeg sample_rate: {value}") from exc
            else:
                sample_rate_overridden = True
        elif key == "channels" and value:
            try:
                channels = int(value)
            except ValueError as exc:
                raise CaptureError(f"Invalid FFmpeg channels: {value}") from exc
            else:
                channels_overridden = True
        elif key == "sample_fmt" and value:
            sample_format = value.lower()
        elif key == "chunk_frames" and value:
            try:
                chunk_frames = max(int(value), 1)
            except ValueError as exc:
                raise CaptureError(f"Invalid FFmpeg chunk_frames: {value}") from exc
        elif key == "chunk_ms" and value:
            try:
                pending_chunk_ms = max(float(value), 0.0)
            except ValueError as exc:
                raise CaptureError(f"Invalid FFmpeg chunk_ms: {value}") from exc
        elif key == "args" and value:
            args_before.extend(shlex.split(value))
        elif key == "out_args" and value:
            args_after.extend(shlex.split(value))
        elif key == "loopback":
            if input_format.lower() != "wasapi":
                raise CaptureError("FFmpeg loopback option is only valid for WASAPI devices")
            normalized_value = (value or "1").strip()
            if normalized_value.lower() in {"true", "yes"}:
                normalized_value = "1"
            if normalized_value.lower() in {"false", "no"}:
                normalized_value = "0"
            if normalized_value not in {"0", "1"}:
                raise CaptureError("FFmpeg loopback value must be 0 or 1")
            input_options.extend(["-loopback", normalized_value])
            loopback_requested = normalized_value == "1"
        elif key.startswith("opt_"):
            option = "-" + key[4:].replace("_", "-")
            if value:
                args_before.extend([option, value])
            else:
                args_before.append(option)
        elif key.startswith("flag_"):
            option = "-" + key[5:].replace("_", "-")
            args_before.append(option)
        elif key.startswith("out_opt_"):
            option = "-" + key[8:].replace("_", "-")
            if value:
                args_after.extend([option, value])
            else:
                args_after.append(option)
        elif key.startswith("out_flag_"):
            option = "-" + key[9:].replace("_", "-")
            args_after.append(option)
        elif not key:
            continue
        else:
            raise CaptureError(f"Unknown FFmpeg device option: {key}")

    if sample_rate <= 0:
        raise CaptureError("FFmpeg sample_rate must be a positive integer")
    if channels <= 0:
        raise CaptureError("FFmpeg channels must be a positive integer")

    if input_format.lower() == "wasapi":
        if loopback_requested and not sample_rate_overridden:
            sample_rate = max(sample_rate, 48000)
        if loopback_requested and not channels_overridden:
            channels = max(channels, 2)

    if pending_chunk_ms is not None and chunk_frames is None:
        chunk_frames = max(int(sample_rate * (pending_chunk_ms / 1000.0)), 1)

    sample_format = sample_format.lower()
    if sample_format not in {"f32le", "s16le", "s32le"}:
        raise CaptureError(
            "FFmpeg output format must be one of: f32le, s16le, s32le"
        )

    return FFmpegDeviceSpec(
        input_format=input_format,
        input_target=input_target,
        args_before_input=args_before,
        input_options=input_options,
        args_after_input=args_after,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        chunk_frames=chunk_frames,
    )


def resolve_ffmpeg_binary(binary: str) -> Optional[str]:
    """Return the absolute path to the requested FFmpeg binary if available."""

    if not binary:
        binary = "ffmpeg"

    # Direct PATH lookup first.
    found = shutil.which(binary)
    if found:
        return found

    candidate = Path(binary)
    if candidate.exists():
        return str(candidate)

    # Windows installers frequently append the .exe suffix even when users omit it.
    if os.name == "nt":  # pragma: no cover - exercised through unit tests via monkeypatch
        suffix = candidate.suffix
        if not suffix.lower().endswith(".exe"):
            exe_candidate = candidate.with_suffix(suffix + ".exe" if suffix else ".exe")
            if exe_candidate.exists():
                return str(exe_candidate)

        exe_name = binary if binary.lower().endswith(".exe") else f"{binary}.exe"
        found_exe = shutil.which(exe_name)
        if found_exe:
            return found_exe

        search_roots = []
        for env_var in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            root = os.environ.get(env_var)
            if root:
                search_roots.append(Path(root) / "ffmpeg" / "bin")
                search_roots.append(Path(root) / "FFmpeg" / "bin")

        search_roots.extend(
            [
                Path("C:/ffmpeg/bin"),
                Path("C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg/bin"),
            ]
        )

        base_name = candidate.name or Path(exe_name).name
        if not base_name.lower().endswith(".exe"):
            base_name = f"{base_name}.exe"

        for directory in search_roots:
            potential = directory / base_name
            if potential.exists():
                return str(potential)

    return None


def ensure_ffmpeg_available(binary: str, *, download_url: Optional[str] = None) -> Optional[str]:
    """Resolve or download the FFmpeg executable.

    If the binary cannot be located via :func:`resolve_ffmpeg_binary`, this helper attempts to
    download a platform-specific build into ``<ADSUM_BASE_DIR>/cache/ffmpeg/<platform>``.
    The download URL is resolved from ``download_url`` or ``ADSUM_FFMPEG_DOWNLOAD_URL`` and
    may include a ``{platform}`` placeholder.
    """

    resolved = resolve_ffmpeg_binary(binary)
    if resolved:
        return resolved

    try:
        from ...config import Settings
    except Exception:
        Settings = None  # type: ignore

    try:
        settings: Optional["Settings"] = get_settings()
    except Exception:  # pragma: no cover - defensive fallback when settings misconfigured
        settings = None

    configured_url = download_url or os.environ.get("ADSUM_FFMPEG_DOWNLOAD_URL")
    if not configured_url and settings is not None:
        configured_url = settings.ffmpeg_download_url

    if not configured_url:
        LOGGER.info(
            "FFmpeg binary '%s' was not found and ADSUM_FFMPEG_DOWNLOAD_URL is not configured.",
            binary,
        )
        return None

    platform = _detect_platform()
    formatted_url = configured_url.format(platform=platform)
    try:
        binary_path = _download_ffmpeg_build(formatted_url, platform, settings)
    except (HTTPError, URLError) as exc:
        LOGGER.error("Failed to download FFmpeg from %s: %s", formatted_url, exc)
        return None
    except Exception:  # pragma: no cover - unexpected extraction or filesystem failures
        LOGGER.exception("Unexpected error while preparing FFmpeg download from %s", formatted_url)
        return None

    if not binary_path:
        LOGGER.error(
            "Downloaded FFmpeg package from %s did not contain an ffmpeg executable.",
            formatted_url,
        )
        return None

    return str(binary_path)


def _download_ffmpeg_build(
    url: str,
    platform: str,
    settings,
) -> Optional[Path]:
    cache_dir = _ffmpeg_cache_dir(platform, settings)

    cached = _locate_ffmpeg_binary(cache_dir)
    if cached:
        LOGGER.debug("Using cached FFmpeg binary at %s", cached)
        return cached

    filename = Path(urlsplit(url).path).name or f"ffmpeg-{platform}"
    download_target = cache_dir / filename

    if not download_target.exists():
        LOGGER.info("Downloading FFmpeg build for %s from %s", platform, url)
        _stream_download(url, download_target)
    else:
        LOGGER.info("Reusing cached FFmpeg download at %s", download_target)

    extracted = _extract_archive(download_target, cache_dir)
    if not extracted:
        binary_path = download_target
        if binary_path.name.lower() not in {"ffmpeg", "ffmpeg.exe"}:
            target_name = "ffmpeg.exe" if platform == "windows" else "ffmpeg"
            destination = cache_dir / target_name
            if destination.exists():
                destination.unlink()
            binary_path = download_target.replace(destination)
        _ensure_executable(binary_path)
        return binary_path

    binary = _locate_ffmpeg_binary(cache_dir)
    if binary:
        _ensure_executable(binary)
    return binary


def _ffmpeg_cache_dir(platform: str, settings) -> Path:
    base_dir: Optional[Path] = None
    if settings is not None:
        base_dir = Path(settings.base_dir)
    elif os.environ.get("ADSUM_BASE_DIR"):
        base_dir = Path(os.environ["ADSUM_BASE_DIR"])
    else:
        base_dir = Path("recordings")

    base_dir = base_dir.expanduser()
    cache_dir = base_dir / "cache" / "ffmpeg" / platform
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _stream_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        status = getattr(response, "status", 200)
        if status and status >= 400:
            raise HTTPError(url, status, getattr(response, "reason", "HTTP error"), None, None)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            shutil.copyfileobj(response, tmp_file)
            tmp_path = Path(tmp_file.name)
    try:
        tmp_path.replace(destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()


def _extract_archive(archive: Path, destination: Path) -> bool:
    if zipfile.is_zipfile(archive):
        LOGGER.info("Extracting FFmpeg archive %s", archive)
        _extract_zip(archive, destination)
        return True
    if tarfile.is_tarfile(archive):
        LOGGER.info("Extracting FFmpeg archive %s", archive)
        _extract_tar(archive, destination)
        return True
    return False


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zip_file:
        base = destination.resolve()
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if not _is_within_directory(base, target):
                raise RuntimeError("Zip archive attempted to write outside the FFmpeg cache")
        zip_file.extractall(destination)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:*") as tar_file:
        base = destination.resolve()
        for member in tar_file.getmembers():
            member_name = member.name or ""
            target = (destination / member_name).resolve()
            if not _is_within_directory(base, target):
                raise RuntimeError("Tar archive attempted to write outside the FFmpeg cache")
        tar_file.extractall(destination)


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory_resolved = directory.resolve()
    except FileNotFoundError:  # pragma: no cover - directory should already exist
        directory_resolved = directory
    try:
        target_resolved = target.resolve()
    except FileNotFoundError:
        target_resolved = target
    return str(target_resolved).startswith(str(directory_resolved))


def _locate_ffmpeg_binary(search_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    if not search_root.exists():
        return None
    for path in search_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == "ffmpeg" or name == "ffmpeg.exe":
            candidates.append(path)

    if not candidates:
        return None

    def _sort_key(value: Path) -> tuple[int, int]:
        suffix_penalty = 0
        if os.name == "nt":
            suffix_penalty = 0 if value.name.lower().endswith(".exe") else 1
        return (suffix_penalty, len(value.parts))

    candidates.sort(key=_sort_key)
    return candidates[0]


def _ensure_executable(path: Path) -> None:
    if os.name != "nt":
        mode = path.stat().st_mode
        path.chmod(mode | 0o111)


__all__ = [
    "FFmpegBinaryNotFoundError",
    "FFmpegDeviceSpec",
    "ensure_ffmpeg_available",
    "parse_ffmpeg_device",
    "resolve_ffmpeg_binary",
]
