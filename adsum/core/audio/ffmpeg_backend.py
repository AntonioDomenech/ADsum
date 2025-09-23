"""Audio capture implementation powered by the FFmpeg command line tool."""

from __future__ import annotations

import contextlib
import io
import queue
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qsl, urlsplit

import numpy as np

from .base import AudioCapture, CaptureError, CaptureInfo
from ...logging import get_logger

LOGGER = get_logger(__name__)


_OUTPUT_CODECS = {
    "f32le": "pcm_f32le",
    "s16le": "pcm_s16le",
    "s32le": "pcm_s32le",
}

_DTYPE_FOR_FORMAT = {
    "f32le": np.float32,
    "s16le": np.int16,
    "s32le": np.int32,
}

_FORMAT_SCALE = {
    "f32le": 1.0,
    "s16le": 32768.0,
    "s32le": 2147483648.0,
}


@dataclass
class FFmpegDeviceSpec:
    """Parsed representation of an FFmpeg capture target."""

    input_format: str
    input_target: str
    args_before_input: List[str]
    args_after_input: List[str]
    sample_rate: int
    channels: int
    sample_format: str
    chunk_frames: Optional[int]


def parse_ffmpeg_device(
    device: str,
    *,
    default_sample_rate: int,
    default_channels: int,
) -> FFmpegDeviceSpec:
    """Return a :class:`FFmpegDeviceSpec` parsed from the user supplied string."""

    if not device:
        raise CaptureError("FFmpeg backend requires a device specification")

    split = urlsplit(device)
    if not split.scheme:
        raise CaptureError(
            "FFmpeg device specification must start with an input format, "
            "for example 'pulse:bluez_source.XX' or 'dshow:audio=Device'",
        )

    input_format = split.scheme
    input_target = (split.netloc + split.path).strip()

    if not input_target:
        raise CaptureError("FFmpeg device specification must include a device identifier")

    args_before: List[str] = []
    args_after: List[str] = []
    sample_rate = int(default_sample_rate)
    channels = int(default_channels)
    sample_format = "f32le"
    chunk_frames: Optional[int] = None
    pending_chunk_ms: Optional[float] = None

    for key, value in parse_qsl(split.query, keep_blank_values=True):
        if key == "sample_rate" and value:
            try:
                sample_rate = int(value)
            except ValueError as exc:
                raise CaptureError(f"Invalid FFmpeg sample_rate: {value}") from exc
        elif key == "channels" and value:
            try:
                channels = int(value)
            except ValueError as exc:
                raise CaptureError(f"Invalid FFmpeg channels: {value}") from exc
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

    if pending_chunk_ms is not None and chunk_frames is None:
        chunk_frames = max(int(sample_rate * (pending_chunk_ms / 1000.0)), 1)

    sample_format = sample_format.lower()
    if sample_format not in _OUTPUT_CODECS:
        raise CaptureError(
            "FFmpeg output format must be one of: "
            + ", ".join(sorted(_OUTPUT_CODECS.keys()))
        )

    return FFmpegDeviceSpec(
        input_format=input_format,
        input_target=input_target,
        args_before_input=args_before,
        args_after_input=args_after,
        sample_rate=sample_rate,
        channels=channels,
        sample_format=sample_format,
        chunk_frames=chunk_frames,
    )


class FFmpegCapture(AudioCapture):
    """Capture stream that reads audio samples from an FFmpeg subprocess."""

    def __init__(
        self,
        info: CaptureInfo,
        *,
        spec: FFmpegDeviceSpec,
        binary: str = "ffmpeg",
        chunk_frames: Optional[int] = None,
    ) -> None:
        self.info = info
        self._spec = spec
        self._binary = binary
        self.info.sample_rate = spec.sample_rate
        self.info.channels = spec.channels
        self._chunk_frames = chunk_frames or spec.chunk_frames or max(self.info.sample_rate // 10, 1)
        self._raw_dtype = _DTYPE_FOR_FORMAT[spec.sample_format]
        self._scale = _FORMAT_SCALE[spec.sample_format]
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._buffer = bytearray()

    def start(self) -> None:
        if self._process is not None:
            return

        executable = _resolve_binary(self._binary)
        if executable is None:
            raise CaptureError(f"FFmpeg binary '{self._binary}' was not found on PATH")

        command = self._build_command(executable)
        LOGGER.info("Starting FFmpeg capture for %s using %s", self.info.name, executable)

        try:
            process = subprocess.Popen(  # noqa: S603 - required to spawn ffmpeg
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise CaptureError(f"Failed to launch FFmpeg binary '{executable}'") from exc

        assert process.stdout is not None  # narrow type for mypy
        assert process.stderr is not None

        self._process = process
        self._stop_event.clear()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            args=(process.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

    def stop(self) -> None:
        process = self._process
        if process is None:
            return

        LOGGER.info("Stopping FFmpeg capture for %s", self.info.name)
        self._stop_event.set()

        with contextlib.suppress(Exception):
            process.terminate()
            process.wait(timeout=1)
    
        if process.poll() is None:
            with contextlib.suppress(Exception):
                process.kill()

    def close(self) -> None:
        process = self._process
        self._process = None
        self._stop_event.set()

        if process is not None:
            with contextlib.suppress(Exception):
                process.stdout and process.stdout.close()
            with contextlib.suppress(Exception):
                process.stderr and process.stderr.close()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1)
            self._reader_thread = None

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
            self._stderr_thread = None

        self._drain_queue()
        self._buffer.clear()

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            if timeout is None or timeout <= 0:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_command(self, executable: str) -> List[str]:
        command: List[str] = [
            executable,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostats",
        ]
        command.extend(self._spec.args_before_input)
        command.extend(["-f", self._spec.input_format])
        command.extend(["-i", self._spec.input_target])
        command.extend(self._spec.args_after_input)
        command.extend(["-vn", "-sn", "-dn"])
        command.extend(["-ac", str(self.info.channels)])
        command.extend(["-ar", str(self.info.sample_rate)])
        command.extend(["-acodec", _OUTPUT_CODECS[self._spec.sample_format]])
        command.extend(["-f", self._spec.sample_format, "pipe:1"])
        return command

    def _reader_loop(self) -> None:  # pragma: no cover - exercised in integration tests
        assert self._process is not None
        stdout = self._process.stdout
        assert stdout is not None
        chunk_bytes = self._chunk_frames * self.info.channels * np.dtype(self._raw_dtype).itemsize

        while not self._stop_event.is_set():
            data = stdout.read(chunk_bytes)
            if not data:
                break
            self._buffer.extend(data)
            self._flush_ready_chunks(chunk_bytes)

        self._flush_ready_chunks(chunk_bytes, drain_all=True)

        if self._process and self._process.poll() not in (0, None):
            LOGGER.warning(
                "FFmpeg exited with code %s while capturing %s",
                self._process.returncode,
                self.info.name,
            )

    def _stderr_loop(self, pipe: io.BufferedReader) -> None:  # pragma: no cover - runtime logging
        try:
            for line in iter(pipe.readline, b""):
                text = line.decode(errors="ignore").strip()
                if text:
                    LOGGER.debug("ffmpeg[%s]: %s", self.info.name, text)
        finally:
            with contextlib.suppress(Exception):
                pipe.close()

    def _flush_ready_chunks(self, chunk_bytes: int, drain_all: bool = False) -> None:
        frame_size = self.info.channels * np.dtype(self._raw_dtype).itemsize
        if frame_size <= 0:
            return

        while len(self._buffer) >= frame_size:
            if len(self._buffer) < chunk_bytes and not drain_all:
                break

            take = min(len(self._buffer), chunk_bytes if len(self._buffer) >= chunk_bytes else len(self._buffer))
            take = (take // frame_size) * frame_size
            if take <= 0:
                break

            raw = bytes(self._buffer[:take])
            del self._buffer[:take]
            if not raw:
                continue

            array = np.frombuffer(raw, dtype=self._raw_dtype)
            if array.size == 0:
                continue

            frames = array.reshape((-1, self.info.channels))
            if self._scale != 1.0:
                frames = frames.astype(np.float32) / float(self._scale)
            else:
                frames = frames.astype(np.float32, copy=False)
            self._queue.put(frames)

        if drain_all:
            self._buffer.clear()

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def _resolve_binary(binary: str) -> Optional[str]:
    """Return the absolute path to the requested FFmpeg binary if available."""

    if not binary:
        binary = "ffmpeg"

    found = shutil.which(binary)
    if found:
        return found

    candidate = Path(binary)
    if candidate.exists():
        return str(candidate)

    return None


__all__ = ["FFmpegCapture", "FFmpegDeviceSpec", "parse_ffmpeg_device"]

