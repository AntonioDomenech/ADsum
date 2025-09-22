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
        self._stream = self._sd.InputStream(
            samplerate=self.info.sample_rate,
            channels=self.info.channels,
            dtype=self._dtype,
            blocksize=self._block_size,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

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


__all__ = ["SoundDeviceCapture"]

