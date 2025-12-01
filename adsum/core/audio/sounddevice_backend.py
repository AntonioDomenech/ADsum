"""Audio capture helpers powered by the sounddevice (PortAudio) library."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import AudioCapture, CaptureError, CaptureInfo
from ...logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class _SoundDeviceConfig:
    device_index: int
    device_name: str
    hostapi_name: str
    max_output_channels: int
    default_samplerate: float


def _prefer_wasapi_device(
    index: int,
    info: dict,
    devices: list[dict],
    host_apis: list[dict],
) -> int:
    """Return an equivalent WASAPI output device index when available."""

    try:
        host_name = host_apis[info["hostapi"]]["name"].lower()
    except Exception:  # pragma: no cover - defensive
        host_name = ""

    if "wasapi" in host_name:
        return index

    name = (info.get("name") or "").strip()
    if not name:
        return index

    for candidate_index, candidate in enumerate(devices):
        candidate_name = (candidate.get("name") or "").strip()
        if candidate_name != name:
            continue
        try:
            candidate_host = host_apis[candidate["hostapi"]]["name"].lower()
        except Exception:
            continue
        if "wasapi" in candidate_host and int(candidate.get("max_output_channels", 0) or 0) > 0:
            return candidate_index

    return index


def _lookup_output_device(
    target: Optional[str],
) -> _SoundDeviceConfig:
    """Return the sounddevice index for the requested output target."""

    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - import failure is surfaced to caller
        raise CaptureError("sounddevice library is unavailable") from exc

    devices = sd.query_devices()
    if not devices:
        raise CaptureError("No audio devices are available via sounddevice")

    host_apis = sd.query_hostapis()

    if target is None or not target.strip() or target.strip().lower() == "default":
        default_pair = sd.default.device
        default_output = None
        if isinstance(default_pair, tuple) and len(default_pair) >= 2:
            default_output = default_pair[1]

        if default_output is None or default_output < 0:
            default_info = sd.query_devices(kind="output")
            default_name = default_info.get("name")
            if default_name:
                for index, info in enumerate(devices):
                    if info.get("max_output_channels", 0) <= 0:
                        continue
                    if info.get("name") == default_name:
                        default_output = index
                        break

        if default_output is None or default_output < 0:
            for index, info in enumerate(devices):
                max_output = info.get("max_output_channels", 0)
                if max_output <= 0:
                    continue
                name = info.get("name", "").lower()
                if "default" in name:
                    default_output = index
                    break

        if default_output is None or default_output < 0:
            alternative = None
            for index, info in enumerate(devices):
                if info.get("max_output_channels", 0) > 0:
                    alternative = index
                    break
            if alternative is None:
                raise CaptureError("Unable to determine the default output device for loopback capture")
            default_output = alternative

        info = devices[default_output]
        preferred_index = _prefer_wasapi_device(default_output, info, devices, host_apis)
        info = devices[preferred_index]
        return _SoundDeviceConfig(
            device_index=preferred_index,
            device_name=info["name"],
            hostapi_name=host_apis[info["hostapi"]]["name"] if host_apis else "unknown",
            max_output_channels=int(info.get("max_output_channels", 0) or 0),
            default_samplerate=float(info.get("default_samplerate", 0.0) or 0.0),
        )

    target_normalized = target.strip().lower()

    # Allow numeric identifiers.
    try:
        numeric_index = int(target_normalized)
    except ValueError:
        numeric_index = None
    if numeric_index is not None and 0 <= numeric_index < len(devices):
        info = devices[numeric_index]
        if info.get("max_output_channels", 0) <= 0:
            raise CaptureError(f"Device index {numeric_index} is not a playback device")
        preferred_index = _prefer_wasapi_device(numeric_index, info, devices, host_apis)
        info = devices[preferred_index]
        return _SoundDeviceConfig(
            device_index=preferred_index,
            device_name=info["name"],
            hostapi_name=host_apis[info["hostapi"]]["name"] if host_apis else "unknown",
            max_output_channels=int(info.get("max_output_channels", 0) or 0),
            default_samplerate=float(info.get("default_samplerate", 0.0) or 0.0),
        )

    candidates: list[tuple[int, dict]] = []
    for index, info in enumerate(devices):
        name = info.get("name", "")
        max_output = int(info.get("max_output_channels", 0) or 0)
        if max_output <= 0:
            continue
        if target_normalized in name.lower():
            candidates.append((index, info))

    if not candidates:
        raise CaptureError(f"No playback device matches '{target}' for loopback capture")

    index, info = candidates[0]
    preferred_index = _prefer_wasapi_device(index, info, devices, host_apis)
    info = devices[preferred_index]
    return _SoundDeviceConfig(
        device_index=preferred_index,
        device_name=info["name"],
        hostapi_name=host_apis[info["hostapi"]]["name"] if host_apis else "unknown",
        max_output_channels=int(info.get("max_output_channels", 0) or 0),
        default_samplerate=float(info.get("default_samplerate", 0.0) or 0.0),
    )


class SoundDeviceLoopbackCapture(AudioCapture):
    """Capture loopback audio using PortAudio via the sounddevice package."""

    def __init__(
        self,
        info: CaptureInfo,
        *,
        target: Optional[str],
        sample_rate: int,
        channels: int,
        chunk_frames: Optional[int] = None,
    ) -> None:
        self.info = info
        self.info.sample_rate = int(sample_rate)
        self.info.channels = int(channels)
        self._target = target
        self._chunk_frames = chunk_frames or max(self.info.sample_rate // 10, 1)

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return

            config = _lookup_output_device(self._target)

            try:
                import sounddevice as sd
            except Exception as exc:  # pragma: no cover - defensive
                raise CaptureError("sounddevice library is unavailable") from exc

            host_name = config.hostapi_name.lower()
            extra_settings = None
            if hasattr(sd, "WasapiSettings") and "wasapi" in host_name:
                try:
                    extra_settings = sd.WasapiSettings(direction="input", exclusive=False, loopback=True)
                except TypeError:
                    try:
                        extra_settings = sd.WasapiSettings(exclusive=False, loopback=True)
                    except TypeError:
                        extra_settings = sd.WasapiSettings(exclusive=False)
                        if hasattr(extra_settings, "loopback"):
                            extra_settings.loopback = True
            elif "wasapi" in host_name:
                LOGGER.warning("sounddevice.WasapiSettings unavailable; loopback may fail for %s", config.device_name)

            dtype = "float32"
            self.info.device = config.device_name

            playback_channels = config.max_output_channels if config.max_output_channels > 0 else self.info.channels
            if playback_channels <= 0:
                playback_channels = self.info.channels if self.info.channels > 0 else 2

            sample_rate_candidates = [
                int(config.default_samplerate) if config.default_samplerate > 0 else None,
                int(self.info.sample_rate) if self.info.sample_rate > 0 else None,
                48000,
                44100,
            ]
            sample_rate_candidates = [rate for rate in sample_rate_candidates if rate and rate > 0]

            channel_candidates = [
                int(playback_channels),
                int(self.info.channels if self.info.channels > 0 else 0),
                8,
                6,
                4,
                2,
                1,
            ]
            channel_candidates = [max(1, ch) for ch in channel_candidates if ch and ch > 0]

            def _unique(values: list[int]) -> list[int]:
                seen: set[int] = set()
                result: list[int] = []
                for value in values:
                    if value not in seen:
                        seen.add(value)
                        result.append(value)
                return result

            sample_rate_candidates = _unique(sample_rate_candidates)
            channel_candidates = _unique(channel_candidates)

            opened_stream = None
            last_error: Optional[Exception] = None

            def _chunk_size(rate: int) -> int:
                return max(rate // 10, 1)

            def _callback(indata, frames, _time, status):  # pragma: no cover - exercised at runtime
                if status:
                    LOGGER.warning("sounddevice status (%s): %s", self.info.name, status)
                if not indata:
                    return
                try:
                    array = np.frombuffer(indata, dtype=np.float32).reshape((-1, self.info.channels))
                except ValueError:
                    inferred_channels = self.info.channels
                    if inferred_channels <= 0:
                        inferred_channels = 1
                    array = np.frombuffer(indata, dtype=np.float32)
                    frames_available = array.size // inferred_channels
                    if frames_available <= 0:
                        return
                    array = array[: frames_available * inferred_channels].reshape((-1, inferred_channels))
                self._queue.put(array)

            for rate in sample_rate_candidates:
                for channels in channel_candidates:
                    channels = max(1, channels)
                    attempts = [
                        (None, config.device_index),
                        config.device_index,
                    ]
                    for device_spec in attempts:
                        try:
                            stream = sd.RawInputStream(
                                samplerate=rate,
                                channels=channels,
                                dtype=dtype,
                                blocksize=_chunk_size(rate),
                                device=device_spec,
                                callback=_callback,
                                extra_settings=extra_settings,
                            )
                        except Exception as exc:
                            last_error = exc
                            LOGGER.debug(
                                "sounddevice loopback failed for %s using device %s with %s Hz, %s ch: %s",
                                config.device_name,
                                device_spec,
                                rate,
                                channels,
                                exc,
                            )
                            continue
                        else:
                            self.info.sample_rate = rate
                            self.info.channels = channels
                            self._chunk_frames = _chunk_size(rate)
                            opened_stream = stream
                            break
                    if opened_stream is not None:
                        break
                if opened_stream is not None:
                    break

            if opened_stream is None:
                if last_error is None:
                    last_error = RuntimeError("Unknown error opening loopback stream")
                raise CaptureError(
                    f"Unable to open loopback stream for '{config.device_name}': {last_error}"
                ) from last_error

            opened_stream.start()
            self._stream = opened_stream
            self._started = True
            LOGGER.info(
                "Started sounddevice loopback capture for %s using %s",
                self.info.name,
                config.device_name,
            )

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            stream = self._stream
            self._stream = None
            self._started = False
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to close sounddevice stream: %s", exc)

    def close(self) -> None:
        self.stop()
        self._drain_queue()

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            data = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return data

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


__all__ = ["SoundDeviceLoopbackCapture"]
