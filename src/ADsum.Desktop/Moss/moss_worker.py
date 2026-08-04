#!/usr/bin/env python3
"""One-shot local MOSS transcription worker for ADsum.

The worker deliberately imports Torch and Transformers only when real model
inference begins.  Its command protocol, WAV chunker, transcript parser,
checkpoint handling, mock engine, and merge logic therefore remain testable on
machines where the multi-gigabyte model runtime is not installed.

Invocation::

    python moss_worker.py --request-file C:\\path\\to\\request.json

When ``--request-file`` is omitted, one JSON object is read from stdin.  Stdout
contains newline-delimited JSON events only; diagnostics belong on stderr.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
import types
import uuid
import wave
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop


PROTOCOL_VERSION = 1
RESULT_SCHEMA_VERSION = 1

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MODEL_REVISION = "e8681d68e7042738ffca8ac8212bc8fcb1131ab8"
UPSTREAM_SOURCE_REVISION = "0e3d1403fd8f1f1c674e883ece96b9f630794ebe"

DEFAULT_CHUNK_SECONDS = 5 * 60
MAX_CHUNK_SECONDS = 30 * 60
DEFAULT_OVERLAP_SECONDS = 30
DEFAULT_MAX_NEW_TOKENS = 4_096
DEFAULT_ENCODER_BATCH_SIZE = 1
DEFAULT_CACHE_MODE = "auto"
MODEL_CONTEXT_TOKENS = 131_072
CONTEXT_SAFETY_TOKENS = 1_024

REQUIRED_SAMPLE_RATE = 16_000
REQUIRED_CHANNELS = 1
REQUIRED_SAMPLE_WIDTH = 2

# ADsum records signed 16-bit PCM.  Treat only digital silence and microscopic
# converter/rounding residue as empty audio.  A peak of 8 is roughly -72 dBFS;
# checking the peak (rather than only average RMS) means that even one sample
# above this deliberately tiny ceiling sends the recording through MOSS.
EFFECTIVE_SILENCE_PEAK = 8

# This is the model author's tested default instruction.  MOSS follows this
# Chinese wording much more reliably than an equivalent English-only prompt,
# even when the recording itself is English, Spanish, Catalan, or multilingual.
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

EventSink = Callable[[dict[str, Any]], None]


class WorkerError(RuntimeError):
    """Expected worker failure that can safely cross the process boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        exit_code: int = 2,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.exit_code = exit_code
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class WavInfo:
    frame_count: int
    sample_rate: int
    channels: int
    sample_width: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "duration": round(self.duration, 6),
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def emit_ndjson(event: dict[str, Any]) -> None:
    """Write one protocol event and flush immediately."""

    payload = {"protocolVersion": PROTOCOL_VERSION, **event}
    print(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default),
        flush=True,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError("invalid_json", f"Could not read JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkerError("invalid_json", "The request must be one JSON object.")
    return payload


def load_request(request_file: str | None) -> dict[str, Any]:
    if request_file:
        return _read_json(Path(request_file).expanduser().resolve())
    raw = sys.stdin.read().lstrip("\ufeff")
    if not raw.strip():
        raise WorkerError("missing_request", "Expected one JSON request on stdin.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkerError("invalid_json", f"Invalid request JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkerError("invalid_json", "The request must be one JSON object.")
    return payload


def default_runtime_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "ADsum" / "MossRuntime"
    return Path.home() / ".adsum" / "MossRuntime"


def default_model_path() -> Path:
    return default_runtime_root() / "Models" / "MOSS" / MODEL_REVISION


def _coerce_number(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise WorkerError("invalid_request", f"{key} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerError("invalid_request", f"{key} must be a number.") from exc
    if not math.isfinite(number):
        raise WorkerError("invalid_request", f"{key} must be finite.")
    return number


def normalize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = payload.get("protocolVersion", PROTOCOL_VERSION)
    if protocol != PROTOCOL_VERSION:
        raise WorkerError(
            "unsupported_protocol",
            f"Unsupported protocolVersion {protocol!r}; expected {PROTOCOL_VERSION}.",
        )

    request_id = str(payload.get("requestId") or uuid.uuid4())
    audio_value = payload.get("audioPath")
    output_value = payload.get("outputPath")
    if not isinstance(audio_value, str) or not audio_value.strip():
        raise WorkerError("invalid_request", "audioPath is required.")
    if not isinstance(output_value, str) or not output_value.strip():
        raise WorkerError("invalid_request", "outputPath is required.")

    audio_path = Path(audio_value).expanduser().resolve()
    output_path = Path(output_value).expanduser().resolve()
    if not audio_path.is_file():
        raise WorkerError("audio_not_found", f"Audio file does not exist: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise WorkerError("unsupported_audio", "The local MOSS worker currently accepts WAV files only.")

    chunk_seconds = _coerce_number(payload, "chunkSeconds", DEFAULT_CHUNK_SECONDS)
    overlap_seconds = _coerce_number(payload, "overlapSeconds", DEFAULT_OVERLAP_SECONDS)
    if chunk_seconds <= 0 or chunk_seconds > MAX_CHUNK_SECONDS:
        raise WorkerError(
            "invalid_chunk_seconds",
            f"chunkSeconds must be greater than 0 and no more than {MAX_CHUNK_SECONDS}.",
        )
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise WorkerError(
            "invalid_overlap_seconds",
            "overlapSeconds must be non-negative and smaller than chunkSeconds.",
        )

    hotwords_value = payload.get("hotwords") or []
    if not isinstance(hotwords_value, list) or not all(isinstance(item, str) for item in hotwords_value):
        raise WorkerError("invalid_hotwords", "hotwords must be an array of strings.")
    hotwords = _clean_hotwords(hotwords_value)

    language = str(payload.get("language") or "auto").strip().lower()
    language_aliases = {
        "auto": "auto",
        "en": "en",
        "english": "en",
        "es": "es",
        "spanish": "es",
        "mixed": "mixed",
        "mixed-en-es": "mixed",
    }
    if language not in language_aliases:
        raise WorkerError("invalid_language", "language must be auto, en, es, or mixed.")
    language = language_aliases[language]

    checkpoint_value = payload.get("checkpointDirectory")
    checkpoint_directory = (
        Path(str(checkpoint_value)).expanduser().resolve()
        if checkpoint_value
        else None
    )

    model_value = payload.get("modelPath")
    model_path = Path(str(model_value)).expanduser().resolve() if model_value else default_model_path()

    mock_config = payload.get("mockInference", False)
    if mock_config is not False and mock_config is not True and not isinstance(mock_config, dict):
        raise WorkerError("invalid_mock_inference", "mockInference must be true, false, or an object.")

    encoder_batch_size = int(payload.get("encoderBatchSize", DEFAULT_ENCODER_BATCH_SIZE))
    if encoder_batch_size < 1 or encoder_batch_size > 16:
        raise WorkerError("invalid_encoder_batch_size", "encoderBatchSize must be from 1 through 16.")

    cache_value = payload.get("cacheMode")
    if cache_value is None and "offloadCache" in payload:
        # Protocol-v1 compatibility for v3.0 callers. Both cache placements
        # produce the same greedy tokens; only their speed and VRAM use differ.
        cache_value = "offloaded" if bool(payload.get("offloadCache")) else "gpu"
    cache_mode = str(cache_value or DEFAULT_CACHE_MODE).strip().lower()
    cache_aliases = {
        "auto": "auto",
        "gpu": "gpu",
        "dynamic": "gpu",
        "on-device": "gpu",
        "offloaded": "offloaded",
        "offload": "offloaded",
        "cpu": "offloaded",
    }
    if cache_mode not in cache_aliases:
        raise WorkerError(
            "invalid_cache_mode",
            "cacheMode must be auto, gpu, or offloaded.",
        )
    cache_mode = cache_aliases[cache_mode]

    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "audioPath": audio_path,
        "outputPath": output_path,
        "checkpointDirectory": checkpoint_directory,
        "language": language,
        "hotwords": hotwords,
        "modelPath": model_path,
        "modelId": MODEL_ID,
        "modelRevision": MODEL_REVISION,
        "chunkSeconds": chunk_seconds,
        "overlapSeconds": overlap_seconds,
        "resume": bool(payload.get("resume", True)),
        "maxNewTokens": int(payload.get("maxNewTokens", DEFAULT_MAX_NEW_TOKENS)),
        "cacheMode": cache_mode,
        "encoderBatchSize": encoder_batch_size,
        "mockInference": mock_config,
        "mockResults": payload.get("mockResults"),
        "voiceRmsThreshold": int(payload.get("voiceRmsThreshold", 200)),
        "coverageSlackSeconds": float(payload.get("coverageSlackSeconds", 8.0)),
    }


def _clean_hotwords(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = " ".join(item.replace("\x00", " ").split())[:80]
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
        if len(result) >= 40:
            break
    return result


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    message = str(error).casefold()
    return "out of memory" in message and ("cuda" in message or "accelerator" in type(error).__name__.casefold())


def build_prompt(language: str, hotwords: Sequence[str]) -> str:
    parts = [DEFAULT_PROMPT]
    if language == "en":
        parts.append("音频主要为英语；请保留原语言，不要翻译。")
    elif language == "es":
        parts.append("音频主要为西班牙语；请保留原语言，不要翻译。")
    elif language == "mixed":
        parts.append("音频可能混合多种语言；请逐段保留原语言，不要翻译。")
    if hotwords:
        parts.append("热词提示：" + ", ".join(hotwords))
    return " ".join(parts)


def inspect_wav(path: Path) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as reader:
            info = WavInfo(
                frame_count=reader.getnframes(),
                sample_rate=reader.getframerate(),
                channels=reader.getnchannels(),
                sample_width=reader.getsampwidth(),
            )
            compression = reader.getcomptype()
    except (OSError, wave.Error) as exc:
        raise WorkerError("invalid_wav", f"Could not read WAV file {path}: {exc}") from exc

    if compression != "NONE":
        raise WorkerError("unsupported_wav", "The WAV file must contain uncompressed PCM audio.")
    if info.sample_rate != REQUIRED_SAMPLE_RATE:
        raise WorkerError(
            "unsupported_wav",
            f"Expected {REQUIRED_SAMPLE_RATE} Hz audio, found {info.sample_rate} Hz.",
        )
    if info.channels != REQUIRED_CHANNELS:
        raise WorkerError(
            "unsupported_wav",
            f"Expected mono audio, found {info.channels} channels.",
        )
    if info.sample_width != REQUIRED_SAMPLE_WIDTH:
        raise WorkerError(
            "unsupported_wav",
            f"Expected 16-bit PCM audio, found {info.sample_width * 8}-bit audio.",
        )
    if info.frame_count <= 0:
        raise WorkerError("empty_audio", "The WAV file contains no audio frames.")
    return info


def wav_is_effectively_silent(
    path: Path,
    *,
    peak_threshold: int = EFFECTIVE_SILENCE_PEAK,
) -> bool:
    """Return true only when every PCM sample is at or below a tiny peak.

    The file is streamed and the scan stops as soon as meaningful signal is
    seen, so normal meetings pay almost no cost.  ``inspect_wav`` is called
    first by ``run_request``; the defensive error conversion here keeps direct
    callers from accidentally treating an unreadable file as silence.
    """

    if peak_threshold < 0:
        raise ValueError("peak_threshold must be non-negative")

    try:
        with wave.open(str(path), "rb") as reader:
            width = reader.getsampwidth()
            block_frames = max(1, reader.getframerate() * 10)
            while data := reader.readframes(block_frames):
                if audioop.max(data, width) > peak_threshold:
                    return False
    except (OSError, wave.Error, audioop.error) as exc:
        raise WorkerError("invalid_wav", f"Could not scan WAV file {path}: {exc}") from exc
    return True


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def build_chunk_plan(
    duration_seconds: float,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[ChunkSpec]:
    if duration_seconds <= 0:
        return []
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be non-negative and smaller than chunk_seconds")

    chunks: list[ChunkSpec] = []
    start = 0.0
    index = 0
    while start < duration_seconds:
        end = min(duration_seconds, start + chunk_seconds)
        chunks.append(ChunkSpec(index=index, start=start, end=end))
        if end >= duration_seconds:
            break
        next_start = end - overlap_seconds
        if next_start <= start:
            raise ValueError("Chunk plan did not make forward progress")
        start = next_start
        index += 1
    return chunks


def extract_wav_chunk(source: Path, destination: Path, chunk: ChunkSpec) -> None:
    """Copy a WAV interval without loading the complete meeting into memory."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(source), "rb") as reader:
            sample_rate = reader.getframerate()
            start_frame = min(reader.getnframes(), max(0, int(round(chunk.start * sample_rate))))
            end_frame = min(reader.getnframes(), max(start_frame, int(round(chunk.end * sample_rate))))
            reader.setpos(start_frame)
            with wave.open(str(destination), "wb") as writer:
                writer.setnchannels(reader.getnchannels())
                writer.setsampwidth(reader.getsampwidth())
                writer.setframerate(sample_rate)
                frames_remaining = end_frame - start_frame
                while frames_remaining > 0:
                    frame_count = min(frames_remaining, sample_rate * 10)
                    data = reader.readframes(frame_count)
                    if not data:
                        break
                    writer.writeframesraw(data)
                    frames_remaining -= len(data) // (reader.getsampwidth() * reader.getnchannels())
                writer.writeframes(b"")
    except (OSError, wave.Error) as exc:
        raise WorkerError("chunk_extraction_failed", f"Could not extract WAV chunk: {exc}") from exc


_SEGMENT_PATTERN = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]"
    r"\[(?P<speaker>S\d+)\]"
    r"(?P<text>.*?)"
    r"\[(?P<end>\d+(?:\.\d+)?)\]"
    r"(?=\s*(?:\[\d+(?:\.\d+)?\]\[S\d+\]|$))",
    re.DOTALL,
)


def normalize_speaker(value: str) -> str:
    match = re.fullmatch(r"S(\d+)", value.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid speaker label: {value!r}")
    return f"S{int(match.group(1)):02d}"


def parse_canonical_transcript(raw_text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for match in _SEGMENT_PATTERN.finditer(raw_text.strip()):
        start = float(match.group("start"))
        end = float(match.group("end"))
        text = " ".join(match.group("text").split())
        if not text:
            continue
        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": normalize_speaker(match.group("speaker")),
                "text": text,
            }
        )
    return segments


def canonical_text(segments: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        f"[{float(segment['start']):.2f}]"
        f"[{normalize_speaker(str(segment['speaker']))}]"
        f"{str(segment['text']).strip()}"
        f"[{float(segment['end']):.2f}]"
        for segment in segments
    )


def _normalized_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def detect_pathological_repetition(segments: Sequence[Mapping[str, Any]]) -> str | None:
    normalized = [_normalized_text(str(segment.get("text", ""))) for segment in segments]
    run = 0
    previous = None
    for text in normalized:
        if text and text == previous:
            run += 1
        else:
            run = 1
            previous = text
        if run >= 3 and len(text) >= 8:
            return "The same segment text was generated at least three times consecutively."

    words = [word for text in normalized for word in text.split()]
    same_word_run = 0
    previous_word = None
    for word in words:
        if word == previous_word:
            same_word_run += 1
        else:
            same_word_run = 1
            previous_word = word
        if same_word_run >= 12:
            return f"The word {word!r} was generated at least twelve times consecutively."

    if len(words) >= 36:
        tail = words[-36:]
        for width in range(4, 13):
            if len(tail) >= width * 3 and tail[-width:] == tail[-2 * width : -width] == tail[-3 * width : -2 * width]:
                return "The generated transcript ends with a phrase repeated at least three times."
    return None


def wav_region_has_voice(
    path: Path,
    start_seconds: float,
    end_seconds: float,
    *,
    rms_threshold: int = 200,
    minimum_voiced_seconds: float = 0.5,
) -> bool:
    if end_seconds <= start_seconds:
        return False
    try:
        with wave.open(str(path), "rb") as reader:
            rate = reader.getframerate()
            width = reader.getsampwidth()
            channels = reader.getnchannels()
            start_frame = min(reader.getnframes(), max(0, int(start_seconds * rate)))
            end_frame = min(reader.getnframes(), max(start_frame, int(end_seconds * rate)))
            reader.setpos(start_frame)
            block_frames = max(1, rate // 4)
            voiced_frames = 0
            frames_remaining = end_frame - start_frame
            while frames_remaining > 0:
                requested = min(frames_remaining, block_frames)
                data = reader.readframes(requested)
                if not data:
                    break
                actual_frames = len(data) // (width * channels)
                if audioop.rms(data, width) >= rms_threshold:
                    voiced_frames += actual_frames
                    if voiced_frames / rate >= minimum_voiced_seconds:
                        return True
                frames_remaining -= actual_frames
    except (OSError, wave.Error, audioop.error):
        return False
    return False


def validate_chunk_result(
    chunk_audio: Path,
    chunk: ChunkSpec,
    raw_text: str,
    segments: Sequence[Mapping[str, Any]],
    *,
    rms_threshold: int = 200,
    coverage_slack_seconds: float = 8.0,
) -> dict[str, Any]:
    previous_start = -1.0
    for position, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise WorkerError(
                "malformed_transcript",
                f"Chunk {chunk.index} contains invalid timestamps at segment {position}.",
                retryable=True,
                details={"chunkIndex": chunk.index},
            )
        if start + 0.01 < previous_start:
            raise WorkerError(
                "malformed_transcript",
                f"Chunk {chunk.index} contains non-monotonic timestamps.",
                retryable=True,
                details={"chunkIndex": chunk.index},
            )
        if end > chunk.duration + 5.0:
            raise WorkerError(
                "malformed_transcript",
                f"Chunk {chunk.index} contains a timestamp beyond the chunk duration.",
                retryable=True,
                details={"chunkIndex": chunk.index},
            )
        previous_start = start

    repetition = detect_pathological_repetition(segments)
    if repetition:
        raise WorkerError(
            "repeated_generation",
            f"Chunk {chunk.index}: {repetition}",
            retryable=True,
            details={"chunkIndex": chunk.index},
        )

    last_end = max((float(segment["end"]) for segment in segments), default=0.0)
    check_from = min(chunk.duration, last_end + coverage_slack_seconds)
    voice_after_result = wav_region_has_voice(
        chunk_audio,
        check_from,
        chunk.duration,
        rms_threshold=rms_threshold,
    )
    if voice_after_result:
        raise WorkerError(
            "incomplete_transcript",
            (
                f"Chunk {chunk.index} ends at {last_end:.2f}s but contains speech later in "
                f"the {chunk.duration:.2f}s audio interval."
            ),
            retryable=True,
            details={
                "chunkIndex": chunk.index,
                "lastTimestamp": round(last_end, 3),
                "chunkDuration": round(chunk.duration, 3),
            },
        )

    if raw_text.strip() and not segments:
        raise WorkerError(
            "malformed_transcript",
            f"Chunk {chunk.index} returned text but no canonical timestamped speaker segments.",
            retryable=True,
            details={"chunkIndex": chunk.index},
        )

    return {
        "lastTimestamp": round(last_end, 6),
        "segmentCount": len(segments),
        "complete": True,
    }


class MockInferenceEngine:
    def __init__(self, request: Mapping[str, Any]) -> None:
        config = request.get("mockInference")
        self.config = dict(config) if isinstance(config, dict) else {}
        if request.get("mockResults") is not None and "chunks" not in self.config:
            self.config["chunks"] = request["mockResults"]

    def infer(self, chunk_audio: Path, chunk: ChunkSpec, prompt: str) -> dict[str, Any]:  # noqa: ARG002
        chunks = self.config.get("chunks") or []
        item: Mapping[str, Any] = chunks[chunk.index] if chunk.index < len(chunks) else {}
        if item.get("error"):
            raise WorkerError(
                str(item.get("code") or "mock_error"),
                str(item["error"]),
                retryable=bool(item.get("retryable", False)),
                details={"chunkIndex": chunk.index},
            )

        supplied_segments = item.get("segments")
        if supplied_segments is None and item.get("rawText") is not None:
            supplied_segments = parse_canonical_transcript(str(item["rawText"]))
        if supplied_segments is None:
            end = max(0.0, chunk.duration - 0.05)
            supplied_segments = [
                {"start": 0.0, "end": end, "speaker": "S01", "text": f"Mock chunk {chunk.index + 1}"}
            ]
        segments = [_normalize_segment(segment) for segment in supplied_segments]
        raw_text = str(item.get("rawText") or canonical_text(segments))
        return {
            "rawText": raw_text,
            "segments": segments,
            "generatedTokens": int(item.get("generatedTokens", max(1, len(raw_text) // 4))),
            "promptTokens": int(item.get("promptTokens", 0)),
            "elapsedSeconds": float(item.get("elapsedSeconds", 0.001)),
            "backend": "mock",
        }


class MossInferenceEngine:
    """Lazily loaded, single-GPU Transformers engine."""

    def __init__(self, request: Mapping[str, Any], emit: EventSink) -> None:
        self.request = request
        self.emit = emit
        self._torch = None
        self._numpy = None
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None
        self._force_offloaded_cache = False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self.emit({"type": "progress", "phase": "loading_model", "progress": 0.05})
        try:
            import numpy as np
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise WorkerError(
                "runtime_missing",
                f"The private MOSS Python runtime is incomplete: {exc}",
                exit_code=20,
            ) from exc

        model_path: Path = self.request["modelPath"]
        if not model_path.is_dir():
            raise WorkerError(
                "model_missing",
                f"The pinned MOSS model snapshot is not installed at {model_path}.",
                exit_code=21,
            )
        if not torch.cuda.is_available():
            raise WorkerError(
                "cuda_unavailable",
                "CUDA is unavailable. ADsum did not load MOSS on the CPU because that would be impractically slow.",
                exit_code=22,
            )

        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
                dtype="auto",
                attn_implementation="sdpa",
            ).to(dtype=dtype).to(device).eval()
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
                fix_mistral_regex=True,
            )
        except Exception as exc:
            raise WorkerError(
                "model_load_failed",
                f"Could not load the pinned MOSS model: {exc}",
                exit_code=21,
            ) from exc

        self._install_memory_efficient_audio_encoder(model, torch)

        self._torch = torch
        self._numpy = np
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype

    def _install_memory_efficient_audio_encoder(self, model: Any, torch: Any) -> None:
        """Encode MOSS's internal 30-second Whisper blocks in small batches.

        Upstream sends all blocks through Whisper at once. A 30-minute input is
        therefore a batch of about 60 blocks, which exhausts an 8 GiB laptop
        GPU even though the model and final audio tokens fit. The VQ adaptor is
        tokenwise, so micro-batching and concatenating its outputs preserves
        the upstream result while bounding temporary encoder activations.
        """

        batch_size = int(self.request.get("encoderBatchSize", DEFAULT_ENCODER_BATCH_SIZE))
        backbone = model.model
        original = backbone.get_audio_features

        def get_audio_features_microbatched(
            _backbone: Any,
            input_features: Any,
            audio_feature_lengths: Any,
            audio_chunk_mapping: Any = None,
        ) -> list[Any]:
            chunk_count = int(input_features.shape[0])
            if chunk_count <= batch_size:
                return original(
                    input_features=input_features,
                    audio_feature_lengths=audio_feature_lengths,
                    audio_chunk_mapping=audio_chunk_mapping,
                )

            if audio_chunk_mapping is None:
                mapping_values = [0] * chunk_count
            else:
                mapping_values = [int(value) for value in audio_chunk_mapping.detach().cpu().tolist()]
            audio_count = max(mapping_values, default=-1) + 1
            outputs: list[Any] = []
            self.emit(
                {
                    "type": "progress",
                    "phase": "encoding_audio_microbatches",
                    "audioChunks": chunk_count,
                    "encoderBatchSize": batch_size,
                }
            )

            for audio_index in range(audio_count):
                source_indices = [
                    index
                    for index, mapped_audio in enumerate(mapping_values)
                    if mapped_audio == audio_index
                ]
                adapted_parts: list[Any] = []
                for offset in range(0, len(source_indices), batch_size):
                    group = source_indices[offset : offset + batch_size]
                    index_tensor = torch.tensor(group, dtype=torch.long, device=input_features.device)
                    group_features = input_features.index_select(0, index_tensor)
                    group_lengths = audio_feature_lengths.index_select(0, index_tensor)
                    local_mapping = torch.zeros(len(group), dtype=torch.long, device=input_features.device)
                    adapted_parts.append(
                        original(
                            input_features=group_features,
                            audio_feature_lengths=group_lengths,
                            audio_chunk_mapping=local_mapping,
                        )[0]
                    )
                if not adapted_parts:
                    raise WorkerError(
                        "invalid_audio_mapping",
                        f"MOSS audio index {audio_index} has no feature chunks.",
                    )
                outputs.append(torch.cat(adapted_parts, dim=1))
            return outputs

        backbone.get_audio_features = types.MethodType(get_audio_features_microbatched, backbone)

    def _load_float_audio(self, path: Path):
        np = self._numpy
        with wave.open(str(path), "rb") as reader:
            raw = reader.readframes(reader.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def _generation_config(self, max_new_tokens: int, cache_mode: str):
        generation_config = copy.deepcopy(self._model.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = False
        # None selects Transformers' ordinary on-device DynamicCache. It is
        # output-equivalent to OffloadedCache but avoids moving every decoder
        # layer's growing K/V tensors across PCIe for every generated token.
        generation_config.cache_implementation = "offloaded" if cache_mode == "offloaded" else None
        return generation_config

    def _generate_once(self, inputs: Mapping[str, Any], max_new_tokens: int, cache_mode: str):
        torch = self._torch
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=self._dtype):
            return self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
                generation_config=self._generation_config(max_new_tokens, cache_mode),
            )

    def _generate_with_cache_fallback(
        self,
        inputs: Mapping[str, Any],
        max_new_tokens: int,
        chunk: ChunkSpec,
    ) -> tuple[Any, str, bool]:
        configured_mode = str(self.request.get("cacheMode", DEFAULT_CACHE_MODE))
        cache_mode = (
            "offloaded"
            if configured_mode == "offloaded" or self._force_offloaded_cache
            else "gpu"
        )
        try:
            return self._generate_once(inputs, max_new_tokens, cache_mode), cache_mode, False
        except Exception as exc:
            if (
                configured_mode != "auto"
                or cache_mode != "gpu"
                or not _is_cuda_out_of_memory(exc)
            ):
                raise

        # We deliberately retry after leaving the exception handler. Python can
        # then release the failed generation traceback and any tensors it kept
        # alive before we ask CUDA to return unused blocks. That gives the
        # lower-memory retry the best chance of succeeding.
        #
        # Keep the exact model, BF16 weights, prompt, and greedy decoding; only
        # move the K/V cache to normal RAM when this GPU is genuinely too full
        # for the fast path. Persist the fallback so later chunks do not waste
        # time repeating an expected OOM.
        self._force_offloaded_cache = True
        self.emit(
            {
                "type": "progress",
                "phase": "cache_fallback_offloaded",
                "chunkIndex": chunk.index,
            }
        )
        gc.collect()
        self._torch.cuda.empty_cache()
        return self._generate_once(inputs, max_new_tokens, "offloaded"), "offloaded", True

    def _infer_loaded(self, chunk_audio: Path, chunk: ChunkSpec, prompt: str) -> dict[str, Any]:
        torch = self._torch
        processor = self._processor
        device = self._device
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        free_before, total_memory = torch.cuda.mem_get_info(device)

        audio_started = time.monotonic()
        audio = self._load_float_audio(chunk_audio)
        audio_load_seconds = time.monotonic() - audio_started
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(chunk_audio)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        processor_started = time.monotonic()
        inputs = processor(
            text=rendered,
            audio=[audio],
            max_length=MODEL_CONTEXT_TOKENS,
            audio_kwargs={"device": str(device)},
            return_tensors="pt",
        ).to(device)
        processor_seconds = time.monotonic() - processor_started
        prompt_tokens = int(inputs["attention_mask"][0].sum().item())
        room = MODEL_CONTEXT_TOKENS - prompt_tokens - CONTEXT_SAFETY_TOKENS
        max_new_tokens = min(int(self.request["maxNewTokens"]), room)
        if max_new_tokens <= 0:
            raise WorkerError(
                "context_exceeded",
                f"Chunk {chunk.index} leaves no room for transcript output.",
                retryable=True,
                details={"chunkIndex": chunk.index, "promptTokens": prompt_tokens},
            )

        generation_started = time.monotonic()
        outputs, cache_mode, cache_fallback = self._generate_with_cache_fallback(
            inputs,
            max_new_tokens,
            chunk,
        )
        torch.cuda.synchronize(device)
        generation_seconds = time.monotonic() - generation_started
        generated_ids = outputs[0][prompt_tokens:]
        raw_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generated_tokens = int(generated_ids.numel())
        segments = parse_canonical_transcript(raw_text)
        elapsed_seconds = time.monotonic() - started
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result = {
            "rawText": raw_text,
            "segments": segments,
            "generatedTokens": generated_tokens,
            "promptTokens": prompt_tokens,
            "elapsedSeconds": elapsed_seconds,
            "audioLoadSeconds": audio_load_seconds,
            "processorSeconds": processor_seconds,
            "generationSeconds": generation_seconds,
            "generatedTokensPerSecond": (
                generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
            ),
            "cacheMode": cache_mode,
            "cacheFallback": cache_fallback,
            "encoderBatchSize": int(self.request["encoderBatchSize"]),
            "freeGpuBeforeMiB": free_before / (1024 * 1024),
            "totalGpuMiB": total_memory / (1024 * 1024),
            "peakGpuAllocatedMiB": peak_allocated / (1024 * 1024),
            "peakGpuReservedMiB": peak_reserved / (1024 * 1024),
            "backend": "transformers",
        }
        del generated_ids, outputs, inputs, audio
        return result

    def infer(self, chunk_audio: Path, chunk: ChunkSpec, prompt: str) -> dict[str, Any]:
        self._ensure_loaded()
        torch = self._torch
        try:
            return self._infer_loaded(chunk_audio, chunk, prompt)
        except WorkerError:
            raise
        except torch.OutOfMemoryError as exc:
            raise WorkerError(
                "cuda_out_of_memory",
                f"MOSS ran out of GPU memory on chunk {chunk.index}.",
                retryable=True,
                exit_code=23,
                details={
                    "chunkIndex": chunk.index,
                    "suggestedChunkSeconds": max(300, int(chunk.duration // 2)),
                },
            ) from exc
        except Exception as exc:
            if _is_cuda_out_of_memory(exc):
                raise WorkerError(
                    "cuda_out_of_memory",
                    f"MOSS ran out of GPU memory on chunk {chunk.index}.",
                    retryable=True,
                    exit_code=23,
                    details={
                        "chunkIndex": chunk.index,
                        "suggestedChunkSeconds": max(300, int(chunk.duration // 2)),
                    },
                ) from exc
            raise WorkerError(
                "inference_failed",
                f"MOSS inference failed on chunk {chunk.index}: {exc}",
                retryable=True,
                details={"chunkIndex": chunk.index},
            ) from exc
        finally:
            try:
                # _infer_loaded's tensor-owning frame has ended, so this now
                # really releases unused cached blocks for the browser/desktop
                # between chunks while keeping the model itself resident.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                # A failed CUDA allocation can leave an asynchronous error on
                # the context. Cleanup must not replace the useful OOM report.
                pass


def _normalize_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "speaker": normalize_speaker(str(segment["speaker"])),
            "text": " ".join(str(segment["text"]).split()),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerError("malformed_transcript", f"Invalid transcript segment: {segment!r}") from exc


def _segment_midpoint(segment: Mapping[str, Any]) -> float:
    return (float(segment["start"]) + float(segment["end"])) / 2.0


def _interval_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))


def _next_global_speaker(used: set[str]) -> str:
    index = 1
    while f"S{index:02d}" in used:
        index += 1
    label = f"S{index:02d}"
    used.add(label)
    return label


def _speaker_mapping(
    existing: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
    overlap_start: float,
    overlap_end: float,
    used_global: set[str],
) -> tuple[dict[str, str], list[str]]:
    local_labels = list(dict.fromkeys(str(segment["speaker"]) for segment in current))
    mapping: dict[str, str] = {}
    uncertain: list[str] = []
    if not local_labels:
        return mapping, uncertain

    previous_overlap = [
        segment
        for segment in existing
        if float(segment["end"]) > overlap_start and float(segment["start"]) < overlap_end
    ]
    current_overlap = [
        segment
        for segment in current
        if float(segment["end"]) > overlap_start and float(segment["start"]) < overlap_end
    ]

    local_duration: dict[str, float] = {}
    for segment in current_overlap:
        label = str(segment["speaker"])
        local_duration[label] = local_duration.get(label, 0.0) + max(
            0.0,
            min(float(segment["end"]), overlap_end) - max(float(segment["start"]), overlap_start),
        )

    scores: list[tuple[float, str, str]] = []
    for current_segment in current_overlap:
        local = str(current_segment["speaker"])
        for previous_segment in previous_overlap:
            global_label = str(previous_segment["speaker"])
            score = _interval_overlap(current_segment, previous_segment)
            if score > 0:
                scores.append((score, local, global_label))

    aggregated: dict[tuple[str, str], float] = {}
    for score, local, global_label in scores:
        key = (local, global_label)
        aggregated[key] = aggregated.get(key, 0.0) + score

    claimed_global: set[str] = set()
    for (local, global_label), score in sorted(aggregated.items(), key=lambda item: item[1], reverse=True):
        if local in mapping or global_label in claimed_global:
            continue
        duration = local_duration.get(local, 0.0)
        if score >= 0.25 and (duration <= 0 or score / duration >= 0.20):
            mapping[local] = global_label
            claimed_global.add(global_label)

    for local in local_labels:
        if local not in mapping:
            mapping[local] = _next_global_speaker(used_global)
            uncertain.append(local)
    return mapping, uncertain


def _deduplicate_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in sorted(segments, key=lambda segment: (float(segment["start"]), float(segment["end"]))):
        segment = dict(raw)
        duplicate_index = None
        normalized = _normalized_text(str(segment["text"]))
        for index in range(max(0, len(result) - 8), len(result)):
            candidate = result[index]
            if normalized != _normalized_text(str(candidate["text"])):
                continue
            overlap = _interval_overlap(segment, candidate)
            shorter = min(
                max(0.001, float(segment["end"]) - float(segment["start"])),
                max(0.001, float(candidate["end"]) - float(candidate["start"])),
            )
            if overlap / shorter >= 0.5:
                duplicate_index = index
                break
        if duplicate_index is None:
            result.append(segment)
        else:
            candidate = result[duplicate_index]
            if float(segment["end"]) - float(segment["start"]) > float(candidate["end"]) - float(candidate["start"]):
                result[duplicate_index] = segment
    return sorted(result, key=lambda segment: (float(segment["start"]), float(segment["end"])))


def merge_chunk_results(
    chunk_results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    merged: list[dict[str, Any]] = []
    warnings: list[str] = []
    mappings: list[dict[str, str]] = []
    used_global: set[str] = set()
    previous_end = 0.0

    for result_position, result in enumerate(chunk_results):
        chunk_index = int(result["index"])
        chunk_start = float(result["start"])
        chunk_end = float(result["end"])
        absolute: list[dict[str, Any]] = []
        for raw_segment in result.get("segments", []):
            segment = _normalize_segment(raw_segment)
            segment["start"] = round(segment["start"] + chunk_start, 6)
            segment["end"] = round(segment["end"] + chunk_start, 6)
            segment["sourceChunk"] = chunk_index
            absolute.append(segment)

        if result_position == 0:
            label_map: dict[str, str] = {}
            for segment in absolute:
                local = str(segment["speaker"])
                if local not in label_map:
                    label_map[local] = _next_global_speaker(used_global)
                segment["speaker"] = label_map[local]
            merged.extend(absolute)
        else:
            overlap_start = chunk_start
            overlap_end = min(previous_end, chunk_end)
            label_map, uncertain = _speaker_mapping(
                merged,
                absolute,
                overlap_start,
                overlap_end,
                used_global,
            )
            for segment in absolute:
                segment["speaker"] = label_map[str(segment["speaker"])]

            if overlap_end > overlap_start:
                seam = (overlap_start + overlap_end) / 2.0
                merged = [segment for segment in merged if _segment_midpoint(segment) < seam]
                absolute = [segment for segment in absolute if _segment_midpoint(segment) >= seam]
            merged.extend(absolute)

            for local in uncertain:
                warnings.append(
                    "Speaker continuity is uncertain in chunk "
                    f"{chunk_index + 1}: local {local} was assigned global {label_map[local]}."
                )

        mappings.append(label_map)
        previous_end = chunk_end

    return _deduplicate_segments(merged), warnings, mappings


def _job_signature(request: Mapping[str, Any], audio_sha256: str, prompt: str) -> str:
    material = {
        "audioSha256": audio_sha256,
        "modelId": MODEL_ID,
        "modelRevision": MODEL_REVISION,
        "prompt": prompt,
        "chunkSeconds": request["chunkSeconds"],
        "overlapSeconds": request["overlapSeconds"],
        "maxNewTokens": request["maxNewTokens"],
        "encoderBatchSize": request["encoderBatchSize"],
        "cacheMode": request["cacheMode"],
        "mock": bool(request["mockInference"]),
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_directory(request: Mapping[str, Any], audio_sha256: str) -> Path:
    explicit = request.get("checkpointDirectory")
    if explicit:
        return Path(explicit)
    audio_path: Path = request["audioPath"]
    output_path: Path = request["outputPath"]
    return output_path.parent / ".moss-checkpoints" / f"{audio_path.stem}-{audio_sha256[:16]}"


def _load_chunk_checkpoint(path: Path, signature: str, chunk: ChunkSpec) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
    except WorkerError:
        return None
    if payload.get("jobSignature") != signature:
        return None
    if int(payload.get("index", -1)) != chunk.index:
        return None
    if abs(float(payload.get("start", -1)) - chunk.start) > 0.001:
        return None
    if abs(float(payload.get("end", -1)) - chunk.end) > 0.001:
        return None
    if not isinstance(payload.get("segments"), list):
        return None
    return payload


def run_request(raw_request: Mapping[str, Any], emit: EventSink = emit_ndjson) -> dict[str, Any]:
    request = normalize_request(raw_request)
    request_id = request["requestId"]
    audio_path: Path = request["audioPath"]
    output_path: Path = request["outputPath"]

    emit({"type": "started", "requestId": request_id, "modelRevision": MODEL_REVISION})
    emit({"type": "progress", "requestId": request_id, "phase": "inspecting_audio", "progress": 0.01})
    wav_info = inspect_wav(audio_path)
    effectively_silent = wav_is_effectively_silent(audio_path)
    audio_sha256 = sha256_file(audio_path)
    prompt = build_prompt(request["language"], request["hotwords"])
    signature = _job_signature(request, audio_sha256, prompt)
    chunks = build_chunk_plan(
        wav_info.duration_seconds,
        request["chunkSeconds"],
        request["overlapSeconds"],
    )
    checkpoint_dir = _checkpoint_directory(request, audio_sha256)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schemaVersion": RESULT_SCHEMA_VERSION,
        "jobSignature": signature,
        "requestId": request_id,
        "audio": {
            "path": str(audio_path),
            "sha256": audio_sha256,
            "durationSeconds": round(wav_info.duration_seconds, 6),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "upstreamSourceRevision": UPSTREAM_SOURCE_REVISION,
        },
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    _atomic_write_json(checkpoint_dir / "manifest.json", manifest)

    if effectively_silent:
        warning = (
            "No speech was transcribed because the recording contains only digital silence "
            "or near-zero PCM samples; MOSS inference was not started."
        )
        emit(
            {
                "type": "progress",
                "requestId": request_id,
                "phase": "silence_detected",
                "progress": 0.92,
            }
        )
        result = {
            "schemaVersion": RESULT_SCHEMA_VERSION,
            "requestId": request_id,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "upstreamSourceRevision": UPSTREAM_SOURCE_REVISION,
                "path": str(request["modelPath"]),
            },
            "audio": {
                "path": str(audio_path),
                "durationSeconds": round(wav_info.duration_seconds, 6),
                "sha256": audio_sha256,
                "sampleRate": wav_info.sample_rate,
                "channels": wav_info.channels,
            },
            "text": "",
            "segments": [],
            "chunks": [],
            "coverage": {
                "complete": True,
                "coveredUntil": round(wav_info.duration_seconds, 6),
                "audioDuration": round(wav_info.duration_seconds, 6),
            },
            "warnings": [warning],
            "checkpointDirectory": str(checkpoint_dir),
        }
        _atomic_write_json(output_path, result)
        emit(
            {
                "type": "completed",
                "requestId": request_id,
                "resultPath": str(output_path),
                "segmentCount": 0,
                "warningCount": 1,
                "warnings": [warning],
            }
        )
        return result

    mock_enabled = (
        request["mockInference"] is not False
        or os.environ.get("ADSUM_MOSS_MOCK_INFERENCE") == "1"
        or os.environ.get("ADSUM_MOSS_MOCK") == "1"
    )
    engine = MockInferenceEngine(request) if mock_enabled else MossInferenceEngine(request, emit)
    chunk_results: list[dict[str, Any]] = []
    resumed_flags: list[bool] = []

    for chunk in chunks:
        checkpoint_path = checkpoint_dir / f"chunk-{chunk.index:04d}.json"
        resumed = False
        checkpoint = (
            _load_chunk_checkpoint(checkpoint_path, signature, chunk)
            if request["resume"]
            else None
        )
        emit(
            {
                "type": "chunk_started",
                "requestId": request_id,
                "index": chunk.index,
                "total": len(chunks),
                "start": round(chunk.start, 3),
                "end": round(chunk.end, 3),
            }
        )
        if checkpoint is not None:
            chunk_result = checkpoint
            resumed = True
        else:
            temporary_path = checkpoint_dir / f".chunk-{chunk.index:04d}-{uuid.uuid4().hex}.wav"
            try:
                extract_wav_chunk(audio_path, temporary_path, chunk)
                inference = engine.infer(temporary_path, chunk, prompt)
                segments = [_normalize_segment(segment) for segment in inference.get("segments", [])]
                validation = validate_chunk_result(
                    temporary_path,
                    chunk,
                    str(inference.get("rawText", "")),
                    segments,
                    rms_threshold=request["voiceRmsThreshold"],
                    coverage_slack_seconds=request["coverageSlackSeconds"],
                )
                chunk_result = {
                    "schemaVersion": RESULT_SCHEMA_VERSION,
                    "jobSignature": signature,
                    "index": chunk.index,
                    "start": round(chunk.start, 6),
                    "end": round(chunk.end, 6),
                    "duration": round(chunk.duration, 6),
                    "rawText": str(inference.get("rawText", "")),
                    "segments": segments,
                    "generatedTokens": int(inference.get("generatedTokens", 0)),
                    "promptTokens": int(inference.get("promptTokens", 0)),
                    "elapsedSeconds": round(float(inference.get("elapsedSeconds", 0.0)), 6),
                    "audioLoadSeconds": round(float(inference.get("audioLoadSeconds", 0.0)), 6),
                    "processorSeconds": round(float(inference.get("processorSeconds", 0.0)), 6),
                    "generationSeconds": round(float(inference.get("generationSeconds", 0.0)), 6),
                    "generatedTokensPerSecond": round(
                        float(inference.get("generatedTokensPerSecond", 0.0)),
                        6,
                    ),
                    "cacheMode": str(inference.get("cacheMode", request["cacheMode"])),
                    "cacheFallback": bool(inference.get("cacheFallback", False)),
                    "encoderBatchSize": int(
                        inference.get("encoderBatchSize", request["encoderBatchSize"])
                    ),
                    "freeGpuBeforeMiB": round(float(inference.get("freeGpuBeforeMiB", 0.0)), 3),
                    "totalGpuMiB": round(float(inference.get("totalGpuMiB", 0.0)), 3),
                    "peakGpuAllocatedMiB": round(
                        float(inference.get("peakGpuAllocatedMiB", 0.0)),
                        3,
                    ),
                    "peakGpuReservedMiB": round(
                        float(inference.get("peakGpuReservedMiB", 0.0)),
                        3,
                    ),
                    "backend": str(inference.get("backend", "unknown")),
                    "validation": validation,
                }
                _atomic_write_json(checkpoint_path, chunk_result)
            finally:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

        chunk_results.append(chunk_result)
        resumed_flags.append(resumed)
        progress = 0.08 + 0.82 * ((chunk.index + 1) / max(1, len(chunks)))
        emit(
            {
                "type": "chunk_completed",
                "requestId": request_id,
                "index": chunk.index,
                "total": len(chunks),
                "resumed": resumed,
                "segmentCount": len(chunk_result.get("segments", [])),
                "progress": round(progress, 6),
            }
        )

    emit({"type": "progress", "requestId": request_id, "phase": "merging", "progress": 0.92})
    segments, warnings, label_mappings = merge_chunk_results(chunk_results)
    chunk_summaries = []
    for chunk_result, label_mapping, resumed in zip(chunk_results, label_mappings, resumed_flags):
        chunk_summaries.append(
            {
                "index": chunk_result["index"],
                "start": chunk_result["start"],
                "end": chunk_result["end"],
                "segmentCount": len(chunk_result.get("segments", [])),
                "generatedTokens": chunk_result.get("generatedTokens", 0),
                "elapsedSeconds": chunk_result.get("elapsedSeconds", 0.0),
                "generationSeconds": chunk_result.get("generationSeconds", 0.0),
                "generatedTokensPerSecond": chunk_result.get("generatedTokensPerSecond", 0.0),
                "cacheMode": chunk_result.get("cacheMode", request["cacheMode"]),
                "cacheFallback": chunk_result.get("cacheFallback", False),
                "encoderBatchSize": chunk_result.get(
                    "encoderBatchSize",
                    request["encoderBatchSize"],
                ),
                "freeGpuBeforeMiB": chunk_result.get("freeGpuBeforeMiB", 0.0),
                "peakGpuAllocatedMiB": chunk_result.get("peakGpuAllocatedMiB", 0.0),
                "peakGpuReservedMiB": chunk_result.get("peakGpuReservedMiB", 0.0),
                "resumed": resumed,
                "labelMapping": label_mapping,
                "checkpointPath": str(checkpoint_dir / f"chunk-{int(chunk_result['index']):04d}.json"),
            }
        )

    covered_until = max((float(segment["end"]) for segment in segments), default=0.0)
    # These totals describe the complete transcript job, including work stored
    # in checkpoints from an earlier run. `resumedChunkCount` below separately
    # makes the amount reused in this invocation explicit.
    total_inference_seconds = sum(
        float(chunk_result.get("elapsedSeconds", 0.0))
        for chunk_result in chunk_results
    )
    total_generation_seconds = sum(
        float(chunk_result.get("generationSeconds", 0.0))
        for chunk_result in chunk_results
    )
    total_generated_tokens = sum(
        int(chunk_result.get("generatedTokens", 0))
        for chunk_result in chunk_results
    )
    result = {
        "schemaVersion": RESULT_SCHEMA_VERSION,
        "requestId": request_id,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "upstreamSourceRevision": UPSTREAM_SOURCE_REVISION,
            "path": str(request["modelPath"]),
        },
        "audio": {
            "path": str(audio_path),
            "durationSeconds": round(wav_info.duration_seconds, 6),
            "sha256": audio_sha256,
            "sampleRate": wav_info.sample_rate,
            "channels": wav_info.channels,
        },
        "text": canonical_text(segments),
        "segments": segments,
        "chunks": chunk_summaries,
        "coverage": {
            "complete": True,
            "coveredUntil": round(covered_until, 6),
            "audioDuration": round(wav_info.duration_seconds, 6),
        },
        "performance": {
            "totalInferenceSeconds": round(total_inference_seconds, 6),
            "totalGenerationSeconds": round(total_generation_seconds, 6),
            "totalGeneratedTokens": total_generated_tokens,
            "generatedTokensPerSecond": round(
                total_generated_tokens / total_generation_seconds
                if total_generation_seconds > 0
                else 0.0,
                6,
            ),
            "audioRealtimeFactor": round(
                total_inference_seconds / wav_info.duration_seconds
                if wav_info.duration_seconds > 0
                else 0.0,
                6,
            ),
            "cacheModes": sorted(
                {
                    str(chunk_result.get("cacheMode", request["cacheMode"]))
                    for chunk_result in chunk_results
                }
            ),
            "cacheFallbackCount": sum(
                1 for chunk_result in chunk_results if bool(chunk_result.get("cacheFallback", False))
            ),
            "resumedChunkCount": sum(1 for resumed in resumed_flags if resumed),
            "peakGpuAllocatedMiB": round(
                max(
                    (float(chunk_result.get("peakGpuAllocatedMiB", 0.0)) for chunk_result in chunk_results),
                    default=0.0,
                ),
                3,
            ),
            "peakGpuReservedMiB": round(
                max(
                    (float(chunk_result.get("peakGpuReservedMiB", 0.0)) for chunk_result in chunk_results),
                    default=0.0,
                ),
                3,
            ),
        },
        "warnings": warnings,
        "checkpointDirectory": str(checkpoint_dir),
    }
    _atomic_write_json(output_path, result)
    emit(
        {
            "type": "completed",
            "requestId": request_id,
            "resultPath": str(output_path),
            "segmentCount": len(segments),
            "warningCount": len(warnings),
            "warnings": warnings,
        }
    )
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local MOSS transcription job.")
    parser.add_argument("--request-file", help="Path to a UTF-8 JSON request. Reads stdin when omitted.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    request_id: str | None = None
    try:
        request = load_request(args.request_file)
        request_id = str(request.get("requestId") or "") or None
        run_request(request)
        return 0
    except WorkerError as exc:
        emit_ndjson(
            {
                "type": "error",
                "requestId": request_id,
                "code": exc.code,
                "message": str(exc),
                "retryable": exc.retryable,
                "details": exc.details,
            }
        )
        return exc.exit_code
    except KeyboardInterrupt:
        emit_ndjson(
            {
                "type": "error",
                "requestId": request_id,
                "code": "cancelled",
                "message": "The MOSS transcription job was cancelled.",
                "retryable": True,
            }
        )
        return 40
    except Exception as exc:  # pragma: no cover - final safety boundary
        print(f"Unexpected MOSS worker failure: {exc!r}", file=sys.stderr, flush=True)
        emit_ndjson(
            {
                "type": "error",
                "requestId": request_id,
                "code": "internal_error",
                "message": str(exc),
                "retryable": False,
            }
        )
        return 50


if __name__ == "__main__":
    raise SystemExit(main())
