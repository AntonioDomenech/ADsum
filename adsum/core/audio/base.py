"""Audio capture abstractions."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CaptureInfo:
    """Metadata about a capture channel."""

    name: str
    sample_rate: int
    channels: int
    device: Optional[str] = None


class AudioCapture(abc.ABC):
    """Abstract capture stream that yields numpy chunks."""

    info: CaptureInfo

    @abc.abstractmethod
    def start(self) -> None:
        """Start the underlying capture stream."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the underlying capture stream."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release all resources associated with the stream."""

    @abc.abstractmethod
    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the next available chunk or ``None`` if none ready."""


class CaptureError(RuntimeError):
    """Raised when audio capture cannot be initialised."""


__all__ = ["AudioCapture", "CaptureError", "CaptureInfo"]

