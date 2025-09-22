"""Audio capture implementation powered by sounddevice/PortAudio."""

from __future__ import annotations

import contextlib
import inspect
import queue
from typing import Optional

import numpy as np

from .base import AudioCapture, CaptureError, CaptureInfo
from ...logging import get_logger

LOGGER = get_logger(__name__)


class SoundDeviceCapture(AudioCapture):
    """Capture stream using the sounddevice library."""

    def __init__(
        self,
        info: CaptureInfo,
        device: Optional[int | str] = None,
        block_size: int = 1024,
        dtype: str = "float32",
    ) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - handled in tests
            raise CaptureError("sounddevice dependency is required for capture") from exc

        self._sd = sd
        self.info = info
        self._device = device
        self._block_size = block_size
        self._dtype = dtype
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._device_info: Optional[dict] = None
        self._extra_settings = None
        self._loopback_channels: Optional[int] = None

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - executed in runtime
        if status:
            LOGGER.warning("sounddevice status: %s", status)
        self._queue.put(indata.copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        LOGGER.info(
            "Starting sounddevice capture for channel %s using device %s",
            self.info.name,
            self._device,
        )

        last_error: Optional[Exception] = None

        requested_sample_rate = int(self.info.sample_rate)

        for channels in self._resolve_channel_candidates():
            for sample_rate in self._resolve_sample_rate_candidates():
                stream_kwargs = dict(
                    samplerate=sample_rate,
                    channels=channels,
                    dtype=self._dtype,
                    blocksize=self._block_size,
                    device=self._device,
                    callback=self._callback,
                )
                if self._extra_settings is not None:
                    stream_kwargs["extra_settings"] = self._extra_settings

                try:
                    stream = self._sd.InputStream(**stream_kwargs)
                except self._sd.PortAudioError as exc:  # pragma: no cover - depends on runtime device
                    last_error = exc
                    message = str(exc)
                    if "Invalid number of channels" in message:
                        LOGGER.warning(
                            "sounddevice rejected %s channel(s) for %s on %s: %s",
                            channels,
                            self.info.name,
                            self._device,
                            message,
                        )
                        break
                    if "sample rate" in message.lower():
                        LOGGER.warning(
                            "sounddevice rejected %s Hz for %s on %s: %s",
                            sample_rate,
                            self.info.name,
                            self._device,
                            message,
                        )
                        continue
                    raise CaptureError(message) from exc

                try:
                    stream.start()
                except self._sd.PortAudioError as exc:  # pragma: no cover - depends on runtime device
                    last_error = exc
                    message = str(exc)
                    with contextlib.suppress(Exception):
                        stream.close()
                    if "Invalid number of channels" in message:
                        LOGGER.warning(
                            "sounddevice rejected %s channel(s) for %s on %s when starting stream: %s",
                            channels,
                            self.info.name,
                            self._device,
                            message,
                        )
                        break
                    if "sample rate" in message.lower() or "host error" in message.lower():
                        LOGGER.warning(
                            "sounddevice failed to start %s at %s Hz on %s: %s",
                            self.info.name,
                            sample_rate,
                            self._device,
                            message,
                        )
                        continue
                    raise CaptureError(message) from exc

                self._stream = stream
                self.info.channels = channels
                if sample_rate != requested_sample_rate:
                    LOGGER.warning(
                        "Adjusted sample rate for %s on %s from %s Hz to %s Hz",
                        self.info.name,
                        self._device,
                        requested_sample_rate,
                        sample_rate,
                    )
                self.info.sample_rate = sample_rate
                LOGGER.info(
                    "Configured %s with %s channel(s) at %s Hz",
                    self.info.name,
                    channels,
                    self.info.sample_rate,
                )
                return

        error_message = (
            f"Failed to open audio stream for {self.info.name} on {self._device}: "
            "No compatible channel/sample rate combination"
        )
        if last_error is not None:
            error_message = f"{error_message} ({last_error})"
        raise CaptureError(error_message) from last_error

    def stop(self) -> None:
        if self._stream is not None:
            LOGGER.info("Stopping capture for %s", self.info.name)
            self._stream.stop()

    def close(self) -> None:
        if self._stream is not None:
            LOGGER.debug("Closing capture stream for %s", self.info.name)
            self._stream.close()
            self._stream = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:  # pragma: no cover - defensive
                break

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            if timeout is None or timeout <= 0:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _resolve_channel_candidates(self) -> list[int]:
        """Return an ordered list of channel counts to try for the device."""

        requested = int(self.info.channels) if self.info.channels else 0
        candidates: list[int] = []

        if requested > 0:
            candidates.append(requested)

        device_info = self._query_device_info()

        max_channels: Optional[int] = None
        if device_info:
            max_channels = int(device_info.get("max_input_channels") or 0)
            if max_channels <= 0 and self._loopback_channels:
                max_channels = self._loopback_channels
            if max_channels <= 0:
                raise CaptureError(f"Device {self._device} does not support input channels")
            if requested > max_channels:
                LOGGER.warning(
                    "Requested %s channel(s) for %s exceeds device capability (%s); using supported maximum",
                    requested,
                    self.info.name,
                    max_channels,
                )
            if max_channels not in candidates:
                candidates.append(max_channels)

        if requested <= 0 and max_channels:
            LOGGER.warning(
                "Requested channel count %s for %s is invalid; defaulting to device capability of %s",
                requested,
                self.info.name,
                max_channels,
            )

        if 1 not in candidates:
            candidates.append(1)

        return [channel for channel in candidates if channel > 0]

    def _resolve_sample_rate_candidates(self) -> list[int]:
        """Return an ordered list of sample rates to try for the device."""

        requested = int(self.info.sample_rate) if self.info.sample_rate else 0
        candidates: list[int] = []

        if requested > 0:
            candidates.append(requested)

        device_info = self._query_device_info()
        default_rate: Optional[int] = None
        if device_info:
            raw = device_info.get("default_samplerate")
            try:
                if raw is not None:
                    default_rate = int(float(raw))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                LOGGER.debug(
                    "Failed to parse default sample rate for %s: %s",
                    self._device,
                    raw,
                )
        if default_rate and default_rate not in candidates:
            candidates.append(default_rate)

        for rate in (48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000):
            if rate not in candidates:
                candidates.append(rate)

        return [rate for rate in candidates if rate > 0]

    def _query_device_info(self) -> Optional[dict]:
        """Return cached device information, querying the backend if necessary."""

        if self._device_info is not None:
            return self._device_info

        try:  # pragma: no cover - depends on runtime availability
            info = self._sd.query_devices(self._device)
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            LOGGER.debug("Failed to query device info for %s: %s", self._device, exc)
            self._device_info = None
            return None

        self._device_info = info
        self._configure_loopback(info)
        return self._device_info

    def _configure_loopback(self, info: dict) -> None:
        if self._loopback_channels:
            return

        max_input = int(info.get("max_input_channels") or 0)
        if max_input > 0:
            return

        hostapi_index = info.get("hostapi")
        hostapi_name = ""
        try:  # pragma: no cover - depends on runtime availability
            if hostapi_index is not None:
                hostapi = self._sd.query_hostapis(int(hostapi_index))
                hostapi_name = str(hostapi.get("name", ""))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to query hostapi info for %s: %s", self._device, exc)

        if "wasapi" not in hostapi_name.lower():
            return

        max_output = int(info.get("max_output_channels") or 0)
        if max_output <= 0:
            return

        settings_factory = getattr(self._sd, "WasapiSettings", None)
        if settings_factory is None:
            raise CaptureError(
                "sounddevice installation does not expose WasapiSettings; "
                "upgrade to a version with WASAPI loopback support"
            )

        try:
            self._extra_settings = _prepare_wasapi_loopback_settings(self._sd)
        except CaptureError:
            raise
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            raise CaptureError(
                f"Failed to configure WASAPI loopback for {self._device}: {exc}"
            ) from exc

        self._loopback_channels = max_output
        LOGGER.info("Configured WASAPI loopback capture for %s", self._device)


def _prepare_wasapi_loopback_settings(sd_module):
    """Return WASAPI settings configured for loopback capture.

    The helper first tries the high-level ``loopback`` keyword introduced in
    recent ``sounddevice`` releases and gracefully falls back to manipulating
    the underlying PortAudio stream info structure when that keyword is
    unavailable.
    """

    settings_factory = getattr(sd_module, "WasapiSettings", None)
    if settings_factory is None:
        raise CaptureError(
            "sounddevice installation does not expose WasapiSettings; "
            "upgrade to a version with WASAPI loopback support"
        )

    if _wasapi_settings_supports_loopback_keyword(settings_factory):
        try:
            return settings_factory(loopback=True)
        except TypeError:
            # Older wheels can report the keyword in their signature but still
            # reject it at runtime. Fall through to the low-level path.
            LOGGER.debug("WasapiSettings rejected loopback keyword; using fallback")
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            raise CaptureError(f"sounddevice failed to enable WASAPI loopback: {exc}") from exc

    if not wasapi_loopback_capable(sd_module):
        raise CaptureError(
            "sounddevice installation does not support enabling WASAPI loopback. "
            "Upgrade to a version that provides WasapiSettings(loopback=...) or exposes "
            "paWinWasapiLoopback"
        )

    try:
        settings = settings_factory()
    except Exception as exc:  # pragma: no cover - defensive
        raise CaptureError(
            f"sounddevice failed to instantiate WasapiSettings for loopback: {exc}"
        ) from exc

    streaminfo = getattr(settings, "_streaminfo", None)
    contents = getattr(streaminfo, "contents", None)
    lib = getattr(sd_module, "_lib", None)
    loopback_flag = getattr(lib, "paWinWasapiLoopback", None) if lib is not None else None
    if contents is None or not hasattr(contents, "flags") or loopback_flag is None:
        raise CaptureError(
            "sounddevice installation does not expose the PaWasapiStreamInfo handle "
            "required to enable loopback capture"
        )

    contents.flags |= loopback_flag
    return settings


def _wasapi_settings_supports_loopback_keyword(settings_factory) -> bool:
    try:
        signature = inspect.signature(settings_factory)
    except (TypeError, ValueError):  # pragma: no cover - builtin or Cython callable
        return False

    for parameter in signature.parameters.values():
        if parameter.name == "loopback" and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def wasapi_loopback_capable(sd_module) -> bool:
    """Return ``True`` when the sounddevice module can configure loopback."""

    settings_factory = getattr(sd_module, "WasapiSettings", None)
    if settings_factory is None:
        return False

    if _wasapi_settings_supports_loopback_keyword(settings_factory):
        return True

    try:
        settings = settings_factory()
    except Exception:  # pragma: no cover - defensive
        return False

    streaminfo = getattr(settings, "_streaminfo", None)
    contents = getattr(streaminfo, "contents", None)
    lib = getattr(sd_module, "_lib", None)
    loopback_flag = getattr(lib, "paWinWasapiLoopback", None) if lib is not None else None
    return bool(contents is not None and hasattr(contents, "flags") and loopback_flag is not None)


__all__ = ["SoundDeviceCapture", "wasapi_loopback_capable"]

