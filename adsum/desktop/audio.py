"""Native Windows audio recording helpers for the v2 desktop app."""

from __future__ import annotations

import math
import queue
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import get_settings
from ..logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DesktopAudioDevice:
    id: str
    name: str
    kind: str
    channels: int
    sample_rate: int
    backend: str
    is_default: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "backend": self.backend,
            "is_default": self.is_default,
            "warning": self.warning,
        }


@dataclass
class RecordingPaths:
    session_dir: Path
    mic_path: Optional[Path]
    system_path: Optional[Path]
    mixed_path: Optional[Path]

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "session_dir": str(self.session_dir),
            "mic_path": str(self.mic_path) if self.mic_path else None,
            "system_path": str(self.system_path) if self.system_path else None,
            "mixed_path": str(self.mixed_path) if self.mixed_path else None,
        }


@dataclass
class TrackMetrics:
    path: Optional[Path]
    duration_seconds: float
    peak: float
    rms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "peak": round(self.peak, 4),
            "rms": round(self.rms, 4),
        }


@dataclass
class RecordingResult:
    name: str
    started_at: float
    stopped_at: float
    paths: RecordingPaths
    metrics: Dict[str, TrackMetrics]

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.stopped_at - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "paths": self.paths.to_dict(),
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
        }


def list_desktop_audio_devices() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "microphones": [device.to_dict() for device in list_microphones()],
        "speakers": [device.to_dict() for device in list_speakers()],
    }


def list_microphones() -> List[DesktopAudioDevice]:
    import sounddevice as sd

    default_input = None
    try:
        candidate = sd.default.device[0]
        if candidate is not None and int(candidate) >= 0:
            default_input = int(candidate)
    except Exception:
        default_input = None

    devices: List[DesktopAudioDevice] = []
    for index, info in enumerate(sd.query_devices()):
        input_channels = int(info.get("max_input_channels") or 0)
        if input_channels <= 0:
            continue
        name = _clean_device_name(str(info.get("name") or f"Input {index}"))
        devices.append(
            DesktopAudioDevice(
                id=str(index),
                name=name,
                kind="microphone",
                channels=input_channels,
                sample_rate=int(float(info.get("default_samplerate") or 48000)),
                backend="sounddevice",
                is_default=index == default_input,
                warning=_bluetooth_warning(name),
            )
        )

    return _dedupe_devices(devices)


def list_speakers() -> List[DesktopAudioDevice]:
    try:
        from ..core.audio.soundcard_backend import _ensure_soundcard_numpy_patch

        _ensure_soundcard_numpy_patch()
        import soundcard as sc
    except Exception as exc:
        LOGGER.warning("Unable to enumerate WASAPI speakers: %s", exc)
        return []

    try:
        default_speaker = sc.default_speaker()
        default_name = getattr(default_speaker, "name", None)
    except Exception:
        default_name = None

    devices: List[DesktopAudioDevice] = []
    for speaker in sc.all_speakers():
        name = _clean_device_name(str(getattr(speaker, "name", "") or "Output"))
        channels = int(getattr(speaker, "channels", None) or 2)
        devices.append(
            DesktopAudioDevice(
                id=name,
                name=name,
                kind="speaker",
                channels=channels,
                sample_rate=48000,
                backend="wasapi-loopback",
                is_default=bool(default_name and name == _clean_device_name(str(default_name))),
                warning=_bluetooth_warning(name),
            )
        )

    return _dedupe_devices(devices)


class NativeRecordingManager:
    """Coordinates microphone and system-audio loopback recording."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.base_dir)
        self._lock = threading.RLock()
        self._active: Optional[_ActiveRecording] = None
        self._last_result: Optional[RecordingResult] = None

    @property
    def last_result(self) -> Optional[RecordingResult]:
        return self._last_result

    def status(self) -> Dict[str, Any]:
        with self._lock:
            if self._active is None:
                return {
                    "state": "idle",
                    "last_result": self._last_result.to_dict() if self._last_result else None,
                }
            return {
                "state": "recording",
                "name": self._active.name,
                "started_at": self._active.started_at,
                "elapsed_seconds": round(time.time() - self._active.started_at, 2),
                "levels": {
                    "microphone": self._active.microphone.level if self._active.microphone else 0.0,
                    "system": self._active.loopback.level if self._active.loopback else 0.0,
                },
                "session_dir": str(self._active.session_dir),
            }

    def start(
        self,
        *,
        name: str,
        microphone_id: Optional[str],
        speaker_id: Optional[str],
    ) -> Dict[str, Any]:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("A recording is already active.")

            session_name = name.strip() or time.strftime("Meeting %Y-%m-%d %H.%M")
            session_dir = self._new_session_dir(session_name)
            session_dir.mkdir(parents=True, exist_ok=True)

            microphone = MicrophoneTrackRecorder(session_dir / "microphone.wav", microphone_id)
            loopback = LoopbackTrackRecorder(session_dir / "system.wav", speaker_id)

            started: List[_BaseTrackRecorder] = []
            try:
                microphone.start()
                started.append(microphone)
                loopback.start()
                started.append(loopback)
            except Exception:
                for recorder in reversed(started):
                    recorder.stop()
                raise

            self._active = _ActiveRecording(
                name=session_name,
                session_dir=session_dir,
                microphone=microphone,
                loopback=loopback,
                started_at=time.time(),
            )
            LOGGER.info("Started desktop recording session %s in %s", session_name, session_dir)
            return self.status()

    def stop(self) -> RecordingResult:
        with self._lock:
            if self._active is None:
                raise RuntimeError("No recording is active.")
            active = self._active
            self._active = None

        active.microphone.stop()
        active.loopback.stop()
        stopped_at = time.time()

        mixed_path = active.session_dir / "mixed.wav"
        source_paths = [active.microphone.path, active.loopback.path]
        if any(path.exists() and path.stat().st_size > 44 for path in source_paths):
            mix_wave_files(source_paths, mixed_path)
        else:
            mixed_path = None  # type: ignore[assignment]

        paths = RecordingPaths(
            session_dir=active.session_dir,
            mic_path=active.microphone.path if active.microphone.path.exists() else None,
            system_path=active.loopback.path if active.loopback.path.exists() else None,
            mixed_path=mixed_path if mixed_path and mixed_path.exists() else None,
        )
        metrics = {
            "microphone": measure_wave_file(paths.mic_path),
            "system": measure_wave_file(paths.system_path),
            "mixed": measure_wave_file(paths.mixed_path),
        }
        result = RecordingResult(
            name=active.name,
            started_at=active.started_at,
            stopped_at=stopped_at,
            paths=paths,
            metrics=metrics,
        )
        self._last_result = result
        LOGGER.info("Stopped desktop recording session %s", active.name)
        return result

    def run_device_test(
        self,
        *,
        microphone_id: Optional[str],
        speaker_id: Optional[str],
        duration_seconds: float = 6.0,
        play_tone: bool = True,
    ) -> RecordingResult:
        if self.status()["state"] == "recording":
            raise RuntimeError("Stop the current recording before running a device test.")

        self.start(
            name="Device test",
            microphone_id=microphone_id,
            speaker_id=speaker_id,
        )
        tone_thread: Optional[threading.Thread] = None
        if play_tone:
            tone_thread = threading.Thread(
                target=_play_test_tone,
                args=(duration_seconds,),
                daemon=True,
            )
            tone_thread.start()
        time.sleep(max(duration_seconds, 1.0))
        result = self.stop()
        if tone_thread:
            tone_thread.join(timeout=1.0)
        return result

    def _new_session_dir(self, name: str) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-").lower()
        slug = slug or "session"
        return self.base_dir / f"{stamp}-{slug}"


@dataclass
class _ActiveRecording:
    name: str
    session_dir: Path
    microphone: "MicrophoneTrackRecorder"
    loopback: "LoopbackTrackRecorder"
    started_at: float


class _BaseTrackRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sample_rate = 48000
        self.channels = 1
        self.level = 0.0
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._writer: Optional[wave.Wave_write] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._writer_thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _open_writer(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = wave.open(str(self.path), "wb")
        self._writer.setnchannels(self.channels)
        self._writer.setsampwidth(2)
        self._writer.setframerate(self.sample_rate)
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

    def _enqueue(self, data: np.ndarray) -> None:
        frames = np.asarray(data, dtype=np.float32)
        if frames.ndim == 1:
            frames = frames.reshape((-1, 1))
        self.level = _rms(frames)
        self._queue.put(frames.copy(), block=False)

    def _write_loop(self) -> None:
        assert self._writer is not None
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frames = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._writer.writeframes(_float_to_pcm16(frames))


class MicrophoneTrackRecorder(_BaseTrackRecorder):
    def __init__(self, path: Path, device_id: Optional[str]) -> None:
        super().__init__(path)
        self.device_id = device_id
        self.device_index: Optional[int] = None
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd

        device_info = _resolve_sounddevice_input(self.device_id)
        self.device_index = device_info["index"]
        self.channels = min(max(int(device_info["channels"]), 1), 2)
        self.sample_rate = int(device_info["sample_rate"])

        self._stop_event.clear()
        self._open_writer()
        self._stream = sd.InputStream(
            device=self.device_index,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                LOGGER.warning("Failed to close microphone stream: %s", exc)
        super().stop()

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            LOGGER.debug("Microphone stream status: %s", status)
        self._enqueue(indata)


class LoopbackTrackRecorder(_BaseTrackRecorder):
    def __init__(self, path: Path, speaker_id: Optional[str]) -> None:
        super().__init__(path)
        self.speaker_id = speaker_id
        self._thread: Optional[threading.Thread] = None
        self._recorder_context: Any = None

    def start(self) -> None:
        from ..core.audio.soundcard_backend import _ensure_soundcard_numpy_patch

        _ensure_soundcard_numpy_patch()
        import soundcard as sc

        speaker = _resolve_soundcard_speaker(sc, self.speaker_id)
        self.channels = min(max(int(getattr(speaker, "channels", None) or 2), 1), 2)
        self.sample_rate = 48000
        microphone = sc.get_microphone(speaker.name, include_loopback=True)
        if microphone is None:
            raise RuntimeError(f"No WASAPI loopback capture endpoint was found for {speaker.name}.")

        self._stop_event.clear()
        self._open_writer()
        self._thread = threading.Thread(
            target=self._record_loop,
            args=(microphone,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        super().stop()

    def _record_loop(self, microphone: Any) -> None:
        blocksize = max(self.sample_rate // 10, 1)
        try:
            with microphone.recorder(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=blocksize,
            ) as recorder:
                while not self._stop_event.is_set():
                    frames = recorder.record(blocksize)
                    if frames is not None and getattr(frames, "size", 0):
                        self._enqueue(np.asarray(frames, dtype=np.float32))
        except Exception as exc:
            LOGGER.exception("System audio loopback capture failed: %s", exc)


def mix_wave_files(paths: List[Path], output_path: Path, *, target_rate: int = 16000) -> None:
    tracks: List[np.ndarray] = []
    for path in paths:
        if not path.exists() or path.stat().st_size <= 44:
            continue
        samples, sample_rate = _read_wave_mono(path)
        if samples.size == 0:
            continue
        tracks.append(_resample(samples, sample_rate, target_rate))

    if not tracks:
        return

    max_length = max(track.shape[0] for track in tracks)
    mixed = np.zeros(max_length, dtype=np.float32)
    for track in tracks:
        padded = np.zeros(max_length, dtype=np.float32)
        padded[: track.shape[0]] = track
        mixed += padded
    mixed = mixed / max(len(tracks), 1)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed / peak * 0.98

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(target_rate)
        writer.writeframes(_float_to_pcm16(mixed.reshape((-1, 1))))


def measure_wave_file(path: Optional[Path]) -> TrackMetrics:
    if path is None or not path.exists() or path.stat().st_size <= 44:
        return TrackMetrics(path=path, duration_seconds=0.0, peak=0.0, rms=0.0)
    samples, sample_rate = _read_wave_mono(path)
    if samples.size == 0 or sample_rate <= 0:
        return TrackMetrics(path=path, duration_seconds=0.0, peak=0.0, rms=0.0)
    return TrackMetrics(
        path=path,
        duration_seconds=float(samples.shape[0]) / sample_rate,
        peak=float(np.max(np.abs(samples))),
        rms=_rms(samples),
    )


def _resolve_sounddevice_input(device_id: Optional[str]) -> Dict[str, Any]:
    import sounddevice as sd

    devices = sd.query_devices()
    index: Optional[int]
    if device_id and device_id not in {"default", "auto"}:
        index = int(device_id)
    else:
        default_candidate = sd.default.device[0]
        index = int(default_candidate) if default_candidate is not None and int(default_candidate) >= 0 else None

    if index is None:
        for candidate, info in enumerate(devices):
            if int(info.get("max_input_channels") or 0) > 0:
                index = candidate
                break

    if index is None:
        raise RuntimeError("No microphone input device is available.")

    info = devices[index]
    return {
        "index": index,
        "channels": int(info.get("max_input_channels") or 1),
        "sample_rate": int(float(info.get("default_samplerate") or 48000)),
    }


def _resolve_soundcard_speaker(sc: Any, speaker_id: Optional[str]) -> Any:
    speakers = sc.all_speakers()
    if not speakers:
        raise RuntimeError("No playback devices are available for WASAPI loopback.")

    if not speaker_id or speaker_id in {"default", "auto"}:
        default_speaker = sc.default_speaker()
        if default_speaker is not None:
            return default_speaker
        return speakers[0]

    normalized = _clean_device_name(speaker_id).lower()
    for speaker in speakers:
        if _clean_device_name(str(speaker.name)).lower() == normalized:
            return speaker
    for speaker in speakers:
        if normalized in _clean_device_name(str(speaker.name)).lower():
            return speaker

    available = ", ".join(_clean_device_name(str(speaker.name)) for speaker in speakers)
    raise RuntimeError(f"Playback device '{speaker_id}' was not found. Available devices: {available}")


def _read_wave_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        width = reader.getsampwidth()
        frames = reader.readframes(reader.getnframes())
    if not frames:
        return np.zeros(0, dtype=np.float32), sample_rate
    if width != 2:
        raise RuntimeError(f"Unsupported sample width in {path}: {width}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)
    return samples.astype(np.float32), sample_rate


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / float(source_rate)
    target_length = max(int(round(duration * target_rate)), 1)
    source_positions = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _float_to_pcm16(frames: np.ndarray) -> bytes:
    values = np.asarray(frames, dtype=np.float32)
    values = np.nan_to_num(values, copy=False)
    values = np.clip(values, -1.0, 1.0)
    return (values * 32767.0).astype("<i2").tobytes()


def _rms(frames: np.ndarray) -> float:
    values = np.asarray(frames, dtype=np.float32)
    if values.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(values)))))


def _clean_device_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\r", " ").replace("\n", " ")).strip()


def _dedupe_devices(devices: List[DesktopAudioDevice]) -> List[DesktopAudioDevice]:
    seen: set[tuple[str, str]] = set()
    unique: List[DesktopAudioDevice] = []
    for device in devices:
        key = (device.kind, device.name.lower())
        if key in seen and not device.is_default:
            continue
        seen.add(key)
        unique.append(device)
    return unique


def _bluetooth_warning(name: str) -> Optional[str]:
    lower = name.lower()
    if "headset" in lower or "hands-free" in lower or "bthhfenum" in lower:
        return (
            "Bluetooth headset microphones can switch Windows into hands-free mode. "
            "If playback quality drops, use a separate mic or an LE Audio capable headset."
        )
    if "bluetooth" in lower or "buds" in lower:
        return "Confirm this output is selected in Windows before recording system audio."
    return None


def _play_test_tone(duration_seconds: float) -> None:
    try:
        import sounddevice as sd

        sample_rate = 48000
        duration = max(duration_seconds - 0.5, 1.0)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = 0.12 * np.sin(2 * np.pi * 660 * t)
        stereo = np.column_stack([tone, tone]).astype(np.float32)
        sd.play(stereo, sample_rate, blocking=True)
        sd.stop()
    except Exception as exc:
        LOGGER.warning("Unable to play loopback test tone: %s", exc)
