"""Audio capture implementation powered by sounddevice/PortAudio."""

from __future__ import annotations

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

        for channels in self._resolve_channel_candidates():
            try:
                stream = self._sd.InputStream(
                    samplerate=self.info.sample_rate,
                    channels=channels,
                    dtype=self._dtype,
                    blocksize=self._block_size,
                    device=self._device,
                    callback=self._callback,
                )
            except self._sd.PortAudioError as exc:  # pragma: no cover - depends on runtime device
                last_error = exc
                message = str(exc)
                if "Invalid number of channels" not in message:
                    raise CaptureError(message) from exc
                LOGGER.warning(
                    "sounddevice rejected %s channel(s) for %s on %s: %s",
                    channels,
                    self.info.name,
                    self._device,
                    message,
                )
                continue

            self._stream = stream
            self.info.channels = channels
            LOGGER.info(
                "Configured %s with %s channel(s) at %s Hz",
                self.info.name,
                channels,
                self.info.sample_rate,
            )
            self._stream.start()
            return

        raise CaptureError(
            f"Failed to open audio stream for {self.info.name} on {self._device}: Invalid number of channels"
        ) from last_error

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

        device_info: Optional[dict] = None
        try:  # pragma: no cover - depends on runtime availability
            device_info = self._sd.query_devices(self._device, "input")
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            LOGGER.debug("Failed to query device info for %s: %s", self._device, exc)

        max_channels: Optional[int] = None
        if device_info:
            max_channels = int(device_info.get("max_input_channels") or 0)
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


__all__ = ["SoundDeviceCapture"]

