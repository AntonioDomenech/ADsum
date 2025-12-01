"""Loopback audio capture powered by the soundcard (WASAPI) library."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import AudioCapture, CaptureError, CaptureInfo
from ...logging import get_logger

LOGGER = get_logger(__name__)

_SOUNDCARD_PATCHED = False


def _ensure_soundcard_numpy_patch() -> None:
    """Patch soundcard's numpy usage for compatibility with NumPy 2.x."""

    global _SOUNDCARD_PATCHED
    if _SOUNDCARD_PATCHED:
        return

    try:
        import soundcard.mediafoundation as _sc_mf  # type: ignore
        import numpy as _np
    except Exception:  # pragma: no cover - optional dependency details
        return

    try:
        current = _sc_mf.numpy.fromstring  # type: ignore[attr-defined]
    except Exception:
        return

    if getattr(current, "__name__", "") == "frombuffer":
        _SOUNDCARD_PATCHED = True
        return

    if hasattr(_np, "frombuffer"):
        try:
            _sc_mf.numpy.fromstring = _np.frombuffer  # type: ignore[attr-defined]
        except Exception:
            return
        else:
            LOGGER.debug("Patched soundcard numpy.fromstring compatibility for NumPy 2.x")
            _SOUNDCARD_PATCHED = True


@dataclass
class _SpeakerSelection:
    speaker: "soundcard.Speaker"
    microphone: "soundcard.Microphone"
    channels: int
    sample_rate: int


def _find_speaker(target: Optional[str]) -> _SpeakerSelection:
    """Resolve the requested speaker into a soundcard object."""

    try:
        _ensure_soundcard_numpy_patch()
        import soundcard as sc
    except Exception as exc:  # pragma: no cover - import failure surfaced to caller
        raise CaptureError("soundcard library is unavailable") from exc

    speakers = sc.all_speakers()
    if not speakers:
        raise CaptureError("No playback devices were detected for WASAPI loopback")

    def _selection_for_speaker(speaker: "sc.Speaker") -> _SpeakerSelection:
        channels = getattr(speaker, "channels", None)
        if not channels or channels <= 0:
            channels = 2
        samplerate = getattr(speaker, "samplerate", None)
        if not samplerate or samplerate <= 0:
            samplerate = 48000
        try:
            microphone = sc.get_microphone(speaker.name, include_loopback=True)
        except Exception as exc:
            raise CaptureError(f"Unable to locate loopback microphone for '{speaker.name}': {exc}") from exc
        if microphone is None:
            raise CaptureError(f"Loopback microphone for '{speaker.name}' is unavailable")
        return _SpeakerSelection(
            speaker=speaker,
            microphone=microphone,
            channels=int(channels),
            sample_rate=int(samplerate),
        )

    if target is None or not target.strip() or target.strip().lower() in {"default", "auto"}:
        speaker = sc.default_speaker()
        if speaker is None:
            raise CaptureError("Unable to determine the default playback device for loopback capture")
        return _selection_for_speaker(speaker)

    normalized = target.strip().lower()

    # Exact name match (case-insensitive).
    for speaker in speakers:
        if speaker.name.strip().lower() == normalized:
            return _selection_for_speaker(speaker)

    # Partial match fallback.
    matches = [
        speaker
        for speaker in speakers
        if normalized in speaker.name.strip().lower()
    ]
    if not matches:
        message_lines = [
            f"Available playback devices: {', '.join(s.name for s in speakers)}",
        ]
        raise CaptureError(
            f"No playback device named '{target}' was found for WASAPI loopback. " + " ".join(message_lines)
        )

    speaker = matches[0]
    return _selection_for_speaker(speaker)


class SoundCardLoopbackCapture(AudioCapture):
    """Capture system audio using WASAPI loopback through soundcard."""

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
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._speaker: Optional["soundcard.Speaker"] = None
        self._microphone: Optional["soundcard.Microphone"] = None
        self._selection: Optional[_SpeakerSelection] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        selection = _find_speaker(self._target)
        self._selection = selection
        self._speaker = selection.speaker
        self._microphone = selection.microphone

        self.info.device = selection.speaker.name
        self.info.channels = max(selection.channels, 1)
        if selection.sample_rate > 0:
            self.info.sample_rate = selection.sample_rate

        self._chunk_frames = max(self.info.sample_rate // 10, 1)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        LOGGER.info(
            "Started soundcard loopback capture for %s using %s",
            self.info.name,
            selection.speaker.name,
        )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def close(self) -> None:
        self.stop()
        self._drain_queue()

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            data = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return data

    def _capture_loop(self) -> None:
        assert self._speaker is not None
        assert self._microphone is not None
        selection = self._selection
        assert selection is not None

        try:
            import soundcard as sc
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("soundcard module became unavailable: %s", exc)
            return

        try:
            recorder = self._microphone.recorder(
                samplerate=self.info.sample_rate,
                channels=self.info.channels,
                blocksize=self._chunk_frames,
            )
        except Exception as exc:
            LOGGER.error("Failed to open soundcard recorder for %s: %s", self.info.device, exc)
            return

        with recorder:
            while not self._stop_event.is_set():
                try:
                    data = recorder.record(self._chunk_frames)
                except Exception as exc:
                    LOGGER.error("soundcard loopback read failed: %s", exc)
                    break
                if data is None or data.size == 0:
                    continue
                try:
                    reshaped = np.asarray(data, dtype=np.float32)
                    if reshaped.ndim == 1:
                        reshaped = reshaped.reshape((-1, self.info.channels))
                except Exception:
                    reshaped = np.array(data, dtype=np.float32, copy=False).reshape((-1, self.info.channels))
                self._queue.put(reshaped, block=False)

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


__all__ = ["SoundCardLoopbackCapture"]
