#!/usr/bin/env python3
"""One-shot, post-recording local speech worker for ADsum.

The worker transcribes one complete, already-saved PCM WAV with
``faster-whisper`` and then runs one whole-recording Community-1 diarization.
It deliberately does not expose a streaming input: callers must finish and
close the recording before starting this process.

The heavy inference libraries are imported only inside the real engines.  As
a result, request validation, checkpointing, speaker assignment, the NDJSON
process protocol, and mock end-to-end jobs remain testable without installing
or downloading either model.

Invocation::

    python local_speech_worker.py --request-file C:\\path\\to\\request.json

When ``--request-file`` is omitted, one JSON request is read from stdin.
Stdout contains newline-delimited JSON events only.  Diagnostics belong on
stderr so the desktop process can parse stdout safely.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROTOCOL_VERSION = 1
RESULT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
WORKER_REVISION = "1"

ASR_MODEL_ID = "openai/whisper-large-v3-turbo"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"

DEFAULT_BATCH_SIZE = 8
BATCH_FALLBACK_ORDER = (8, 4, 2)
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "int8_float16"

EventSink = Callable[[dict[str, Any]], None]

# On Windows, CTranslate2 and PyTorch ship CUDA DLLs in different package
# folders. ``os.add_dll_directory`` handles must remain alive for the lifetime
# of the process or Windows removes those search locations again.
_DLL_DIRECTORY_HANDLES: list[Any] = []


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


class ProgressReporter:
    """Attach consistent wall-clock numbers to progress events."""

    def __init__(
        self,
        request_id: str,
        emit: EventSink,
        started: float,
    ) -> None:
        self.request_id = request_id
        self.emit = emit
        self.started = started

    def progress(
        self,
        phase: str,
        progress: float,
        *,
        stage_started: float | None = None,
        eta_seconds: float | None = None,
        **extra: Any,
    ) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self.started)
        event: dict[str, Any] = {
            "type": "progress",
            "requestId": self.request_id,
            "phase": phase,
            "progress": round(min(1.0, max(0.0, progress)), 6),
            "elapsedSeconds": round(elapsed, 6),
        }
        if stage_started is not None:
            event["stageElapsedSeconds"] = round(max(0.0, now - stage_started), 6)
        if eta_seconds is not None and math.isfinite(eta_seconds):
            event["etaSeconds"] = round(max(0.0, eta_seconds), 6)
        event.update(extra)
        self.emit(event)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def emit_ndjson(event: dict[str, Any]) -> None:
    payload = {"protocolVersion": PROTOCOL_VERSION, **event}
    print(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default),
        flush=True,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
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


def default_asr_model_path() -> Path:
    configured = os.environ.get("ADSUM_LOCAL_SPEECH_ASR_MODEL")
    if configured:
        return Path(configured).expanduser().resolve()
    return default_runtime_root() / "Models" / "FasterWhisper" / "large-v3-turbo"


def default_diarization_model_path() -> Path:
    configured = os.environ.get("ADSUM_LOCAL_SPEECH_DIARIZATION_MODEL")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        default_runtime_root()
        / "Models"
        / "Pyannote"
        / "speaker-diarization-community-1"
    )


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


def _coerce_optional_positive_integer(
    payload: Mapping[str, Any],
    key: str,
) -> int | None:
    value = payload.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise WorkerError("invalid_request", f"{key} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkerError("invalid_request", f"{key} must be a positive integer.") from exc
    if parsed <= 0:
        raise WorkerError("invalid_request", f"{key} must be a positive integer.")
    return parsed


def _environment_boolean(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


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
    if audio_path.suffix.casefold() != ".wav":
        raise WorkerError("unsupported_audio", "The local speech worker accepts WAV files only.")

    if payload.get("recordingComplete", True) is not True:
        raise WorkerError(
            "recording_not_complete",
            "The recording must be stopped and the WAV must be complete before transcription starts.",
            retryable=True,
        )

    hotwords_value = payload.get("hotwords") or []
    if not isinstance(hotwords_value, list) or not all(
        isinstance(item, str) for item in hotwords_value
    ):
        raise WorkerError("invalid_hotwords", "hotwords must be an array of strings.")

    language_value = str(payload.get("language") or "auto").strip().casefold()
    language_aliases = {
        "auto": "auto",
        "mixed": "auto",
        "mixed-en-es": "auto",
        "english": "en",
        "spanish": "es",
        "catalan": "ca",
    }
    language = language_aliases.get(language_value, language_value)
    if language != "auto" and not (
        2 <= len(language) <= 3 and language.isascii() and language.isalpha()
    ):
        raise WorkerError(
            "invalid_language",
            "language must be auto, mixed, or a two/three-letter language code.",
        )

    mock_config = payload.get("mockInference", False)
    if mock_config is not False and mock_config is not True and not isinstance(mock_config, dict):
        raise WorkerError("invalid_mock_inference", "mockInference must be true, false, or an object.")
    mock_enabled = mock_config is not False or _environment_boolean(
        "ADSUM_LOCAL_SPEECH_MOCK_INFERENCE"
    )

    batch_value = payload.get(
        "batchSize",
        os.environ.get("ADSUM_LOCAL_SPEECH_BATCH_SIZE", DEFAULT_BATCH_SIZE),
    )
    try:
        batch_size = int(batch_value)
    except (TypeError, ValueError) as exc:
        raise WorkerError("invalid_batch_size", "batchSize must be 8, 4, or 2.") from exc
    if batch_size not in BATCH_FALLBACK_ORDER:
        raise WorkerError("invalid_batch_size", "batchSize must be 8, 4, or 2.")

    device = str(
        payload.get("device")
        or os.environ.get("ADSUM_LOCAL_SPEECH_DEVICE")
        or DEFAULT_DEVICE
    ).strip().casefold()
    if device not in {"cuda", "cpu", "auto"}:
        raise WorkerError("invalid_device", "device must be cuda, cpu, or auto.")
    compute_type = str(
        payload.get("computeType")
        or os.environ.get("ADSUM_LOCAL_SPEECH_COMPUTE_TYPE")
        or DEFAULT_COMPUTE_TYPE
    ).strip()
    if not compute_type:
        raise WorkerError("invalid_compute_type", "computeType cannot be empty.")

    if payload.get("wordTimestamps", True) is not True:
        raise WorkerError(
            "word_timestamps_required",
            "wordTimestamps must remain true because speaker assignment happens per word.",
        )

    exact_speakers = _coerce_optional_positive_integer(payload, "numSpeakers")
    minimum_speakers = _coerce_optional_positive_integer(payload, "minSpeakers")
    maximum_speakers = _coerce_optional_positive_integer(payload, "maxSpeakers")
    if (
        minimum_speakers is not None
        and maximum_speakers is not None
        and minimum_speakers > maximum_speakers
    ):
        raise WorkerError("invalid_request", "minSpeakers cannot exceed maxSpeakers.")

    asr_model_value = payload.get("asrModelPath") or payload.get("modelPath")
    diarization_model_value = payload.get("diarizationModelPath")
    checkpoint_value = payload.get("checkpointDirectory")

    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": request_id,
        "audioPath": audio_path,
        "outputPath": output_path,
        "checkpointDirectory": (
            Path(str(checkpoint_value)).expanduser().resolve() if checkpoint_value else None
        ),
        "language": language,
        "hotwords": _clean_hotwords(hotwords_value),
        "asrModelPath": (
            Path(str(asr_model_value)).expanduser().resolve()
            if asr_model_value
            else default_asr_model_path()
        ),
        "diarizationModelPath": (
            Path(str(diarization_model_value)).expanduser().resolve()
            if diarization_model_value
            else default_diarization_model_path()
        ),
        "batchSize": batch_size,
        "device": device,
        "computeType": compute_type,
        "vadFilter": bool(payload.get("vadFilter", True)),
        "wordTimestamps": True,
        "numSpeakers": exact_speakers,
        "minSpeakers": minimum_speakers,
        "maxSpeakers": maximum_speakers,
        "resume": bool(payload.get("resume", True)),
        "mockInference": dict(mock_config) if isinstance(mock_config, dict) else {},
        "mockEnabled": mock_enabled,
    }


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
    if info.frame_count <= 0 or info.sample_rate <= 0:
        raise WorkerError("empty_audio", "The WAV file contains no audio frames.")
    if info.sample_width != 2:
        raise WorkerError(
            "unsupported_wav",
            f"Expected signed 16-bit PCM audio, found {info.sample_width * 8}-bit audio.",
        )
    if info.channels <= 0:
        raise WorkerError("unsupported_wav", "The WAV file has no audio channels.")
    return info


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(block_size):
                digest.update(block)
    except OSError as exc:
        raise WorkerError("audio_read_failed", f"Could not read audio file {path}: {exc}") from exc
    return digest.hexdigest()


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".casefold()
    return (
        "out of memory" in text
        and ("cuda" in text or "gpu" in text or "ctranslate" in text)
    ) or "cuda_error_out_of_memory" in text


def _cleanup_accelerator() -> None:
    gc.collect()
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    except Exception:
        # Cleanup must not hide the useful inference error that came first.
        pass


def _prepare_ctranslate2_windows_dlls() -> None:
    """Expose PyTorch's bundled CUDA DLL folder before importing CTranslate2.

    Finding the package spec does not execute ``torch.__init__`` and therefore
    does not create a PyTorch CUDA context while faster-whisper owns the GPU.
    Community-1 imports Torch later, after the CTranslate2 model is destroyed.
    """

    if os.name != "nt":
        return
    try:
        torch_spec = importlib.util.find_spec("torch")
    except (ImportError, AttributeError, ValueError):
        return
    if torch_spec is None:
        return
    candidates: list[Path] = []
    if torch_spec.submodule_search_locations:
        candidates.extend(
            Path(location) / "lib" for location in torch_spec.submodule_search_locations
        )
    if torch_spec.origin:
        candidates.append(Path(torch_spec.origin).resolve().parent / "lib")
    for library_directory in dict.fromkeys(candidates):
        if not library_directory.is_dir():
            continue
        directory_text = str(library_directory.resolve())
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if directory_text.casefold() not in {entry.casefold() for entry in path_entries}:
            os.environ["PATH"] = (
                directory_text
                if not os.environ.get("PATH")
                else directory_text + os.pathsep + os.environ["PATH"]
            )
        add_directory = getattr(os, "add_dll_directory", None)
        if add_directory is not None:
            try:
                _DLL_DIRECTORY_HANDLES.append(add_directory(directory_text))
            except OSError:
                # PATH is still updated; the eventual import error will name
                # the missing DLL if Windows rejects this directory handle.
                pass


def _read_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkerError("invalid_inference_output", f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise WorkerError("invalid_inference_output", f"{name} must be finite.")
    return number


def _normalize_asr_word(value: Any) -> dict[str, Any] | None:
    text = str(_read_attr(value, "word", _read_attr(value, "text", "")))
    start_value = _read_attr(value, "start")
    end_value = _read_attr(value, "end")
    if start_value is None or end_value is None or not text.strip():
        return None
    start = _finite_float(start_value, name="word.start")
    end = _finite_float(end_value, name="word.end")
    if start < 0 or end < start:
        return None
    probability_value = _read_attr(value, "probability")
    probability = None
    if probability_value is not None:
        parsed_probability = _finite_float(probability_value, name="word.probability")
        probability = min(1.0, max(0.0, parsed_probability))
    return {
        "start": round(start, 6),
        "end": round(end, 6),
        "text": text,
        "probability": round(probability, 6) if probability is not None else None,
    }


def _normalize_asr_segment(value: Any, index: int) -> dict[str, Any] | None:
    text = " ".join(str(_read_attr(value, "text", "")).split())
    start_value = _read_attr(value, "start")
    end_value = _read_attr(value, "end")
    if start_value is None or end_value is None or not text:
        return None
    start = _finite_float(start_value, name="segment.start")
    end = _finite_float(end_value, name="segment.end")
    if start < 0 or end < start:
        return None
    raw_words = _read_attr(value, "words", None) or []
    words = [word for item in raw_words if (word := _normalize_asr_word(item)) is not None]
    return {
        "id": int(_read_attr(value, "id", index)),
        "start": round(start, 6),
        "end": round(end, 6),
        "text": text,
        "words": words,
    }


class MockAsrEngine:
    def __init__(self, request: Mapping[str, Any], wav_info: WavInfo) -> None:
        self.config = dict(request.get("mockInference") or {})
        self.wav_info = wav_info

    def load(self) -> None:
        return None

    def transcribe(self, batch_size: int) -> dict[str, Any]:
        oom_batches = {int(value) for value in self.config.get("oomBatchSizes", [])}
        if batch_size in oom_batches:
            raise RuntimeError(f"CUDA out of memory in mock batch {batch_size}")
        supplied = self.config.get("asrSegments")
        if supplied is None:
            duration = self.wav_info.duration_seconds
            supplied = [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": duration,
                    "text": "Mock transcription",
                    "words": [
                        {
                            "start": 0.0,
                            "end": duration,
                            "word": "Mock transcription",
                            "probability": 1.0,
                        }
                    ],
                }
            ]
        segments = [
            segment
            for index, item in enumerate(supplied)
            if (segment := _normalize_asr_segment(item, index)) is not None
        ]
        return {
            "segments": segments,
            "detectedLanguage": str(self.config.get("detectedLanguage", "en")),
            "languageProbability": float(self.config.get("languageProbability", 1.0)),
            "durationAfterVadSeconds": float(
                self.config.get("durationAfterVadSeconds", self.wav_info.duration_seconds)
            ),
            "backend": "mock",
        }

    def unload(self) -> None:
        return None


class FasterWhisperEngine:
    """Lazily loaded CTranslate2 batched ASR engine."""

    def __init__(self, request: Mapping[str, Any]) -> None:
        self.request = request
        self.model: Any = None
        self.pipeline: Any = None

    def load(self) -> None:
        model_path: Path = self.request["asrModelPath"]
        if not model_path.is_dir():
            raise WorkerError(
                "asr_setup_required",
                f"The local faster-whisper large-v3-turbo model is not installed at {model_path}.",
                exit_code=21,
                details={
                    "expectedPath": str(model_path),
                    "modelId": ASR_MODEL_ID,
                    "action": "Run ADsum local speech model setup before transcribing.",
                },
            )
        _prepare_ctranslate2_windows_dlls()
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ImportError as exc:
            raise WorkerError(
                "runtime_missing",
                f"The local speech Python runtime is incomplete: {exc}",
                exit_code=20,
                details={"package": "faster-whisper"},
            ) from exc
        try:
            self.model = WhisperModel(
                str(model_path),
                device=str(self.request["device"]),
                compute_type=str(self.request["computeType"]),
                local_files_only=True,
            )
            self.pipeline = BatchedInferencePipeline(model=self.model)
        except Exception as exc:
            if _is_cuda_out_of_memory(exc):
                raise WorkerError(
                    "cuda_out_of_memory",
                    "The ASR model could not fit in GPU memory.",
                    retryable=True,
                    exit_code=23,
                ) from exc
            raise WorkerError(
                "asr_model_load_failed",
                f"Could not load the local faster-whisper model: {exc}",
                exit_code=21,
            ) from exc

    def transcribe(self, batch_size: int) -> dict[str, Any]:
        if self.pipeline is None:
            raise WorkerError("internal_error", "The ASR engine was not loaded.", exit_code=50)
        language = None if self.request["language"] == "auto" else self.request["language"]
        options: dict[str, Any] = {
            "batch_size": batch_size,
            "language": language,
            "vad_filter": bool(self.request["vadFilter"]),
            "vad_parameters": {"min_silence_duration_ms": 500},
            "word_timestamps": True,
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "without_timestamps": False,
            "multilingual": language is None,
        }
        hotwords = self.request.get("hotwords") or []
        if hotwords:
            options["hotwords"] = ", ".join(str(item) for item in hotwords)
        try:
            segment_iterator, info = self.pipeline.transcribe(
                str(self.request["audioPath"]),
                **options,
            )
            raw_segments = list(segment_iterator)
        except Exception:
            # The caller owns OOM classification so it can retry at 8 -> 4 -> 2.
            raise
        segments = [
            segment
            for index, item in enumerate(raw_segments)
            if (segment := _normalize_asr_segment(item, index)) is not None
        ]
        language_probability = _read_attr(info, "language_probability")
        duration_after_vad = _read_attr(info, "duration_after_vad")
        return {
            "segments": segments,
            "detectedLanguage": _read_attr(info, "language"),
            "languageProbability": (
                float(language_probability) if language_probability is not None else None
            ),
            "durationAfterVadSeconds": (
                float(duration_after_vad) if duration_after_vad is not None else None
            ),
            "backend": "faster-whisper",
        }

    def unload(self) -> None:
        self.pipeline = None
        self.model = None
        _cleanup_accelerator()


def _batch_attempt_order(initial_batch_size: int) -> list[int]:
    start = BATCH_FALLBACK_ORDER.index(initial_batch_size)
    return list(BATCH_FALLBACK_ORDER[start:])


def _run_asr_with_fallback(
    engine: MockAsrEngine | FasterWhisperEngine,
    request: Mapping[str, Any],
    reporter: ProgressReporter,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    attempts: list[dict[str, Any]] = []
    order = _batch_attempt_order(int(request["batchSize"]))
    for position, batch_size in enumerate(order):
        attempt_started = time.monotonic()
        reporter.progress(
            "transcribing_audio",
            0.15,
            stage_started=attempt_started,
            batchSize=batch_size,
            attempt=position + 1,
            attemptCount=len(order),
        )
        try:
            result = engine.transcribe(batch_size)
        except Exception as exc:
            elapsed = time.monotonic() - attempt_started
            attempts.append(
                {
                    "batchSize": batch_size,
                    "elapsedSeconds": round(elapsed, 6),
                    "succeeded": False,
                    "cudaOutOfMemory": _is_cuda_out_of_memory(exc),
                }
            )
            if not _is_cuda_out_of_memory(exc):
                if isinstance(exc, WorkerError):
                    raise
                raise WorkerError(
                    "asr_inference_failed",
                    f"Local ASR failed: {exc}",
                    retryable=True,
                    details={"batchSize": batch_size},
                ) from exc
            if position + 1 >= len(order):
                raise WorkerError(
                    "cuda_out_of_memory",
                    "Local ASR ran out of GPU memory even at batch size 2.",
                    retryable=True,
                    exit_code=23,
                    details={"attemptedBatchSizes": order},
                ) from exc
            next_batch_size = order[position + 1]
            _cleanup_accelerator()
            reporter.progress(
                "batch_size_fallback",
                0.15,
                failedBatchSize=batch_size,
                nextBatchSize=next_batch_size,
                attemptElapsedSeconds=round(elapsed, 6),
            )
            continue
        elapsed = time.monotonic() - attempt_started
        attempts.append(
            {
                "batchSize": batch_size,
                "elapsedSeconds": round(elapsed, 6),
                "succeeded": True,
                "cudaOutOfMemory": False,
            }
        )
        return result, attempts, batch_size
    raise WorkerError("internal_error", "ASR fallback order was empty.", exit_code=50)


def _load_pcm_for_pyannote(path: Path, torch_module: Any, numpy_module: Any) -> dict[str, Any]:
    """Load ADsum's PCM WAV without torchcodec/FFmpeg decoder dependencies."""

    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            raw = reader.readframes(reader.getnframes())
    except (OSError, wave.Error) as exc:
        raise WorkerError("audio_read_failed", f"Could not load WAV for diarization: {exc}") from exc
    if sample_width != 2:
        raise WorkerError("unsupported_wav", "Diarization requires signed 16-bit PCM WAV audio.")
    samples = numpy_module.frombuffer(raw, dtype="<i2").astype(numpy_module.float32)
    samples /= 32768.0
    try:
        samples = samples.reshape(-1, channels).transpose()
    except ValueError as exc:
        raise WorkerError("invalid_wav", "The PCM WAV contains an incomplete sample frame.") from exc
    waveform = torch_module.from_numpy(samples)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _extract_annotation_turns(annotation: Any) -> list[dict[str, Any]]:
    if annotation is None:
        return []
    try:
        tracks = annotation.itertracks(yield_label=True)
    except AttributeError as exc:
        raise WorkerError(
            "diarization_output_incompatible",
            "Community-1 did not return a pyannote Annotation.",
        ) from exc
    turns: list[dict[str, Any]] = []
    for turn, _track, speaker in tracks:
        start = float(turn.start)
        end = float(turn.end)
        if math.isfinite(start) and math.isfinite(end) and end > start:
            turns.append({"start": start, "end": end, "speaker": str(speaker)})
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return turns


class MockDiarizationEngine:
    def __init__(self, request: Mapping[str, Any], wav_info: WavInfo) -> None:
        self.config = dict(request.get("mockInference") or {})
        self.wav_info = wav_info

    def load(self) -> None:
        return None

    def diarize(self) -> dict[str, Any]:
        exclusive = self.config.get("exclusiveSpeakerTurns")
        if exclusive is None:
            exclusive = [
                {
                    "start": 0.0,
                    "end": self.wav_info.duration_seconds,
                    "speaker": "mock-speaker-1",
                }
            ]
        regular = self.config.get("regularSpeakerTurns", exclusive)
        return {
            "exclusiveTurns": _normalize_raw_turns(exclusive),
            "regularTurns": _normalize_raw_turns(regular),
            "backend": "mock",
            "audioLoadSeconds": 0.0,
        }

    def unload(self) -> None:
        return None


class PyannoteDiarizationEngine:
    """Whole-recording Community-1 diarization using preloaded PCM audio."""

    def __init__(self, request: Mapping[str, Any]) -> None:
        self.request = request
        self.pipeline: Any = None
        self.torch: Any = None
        self.numpy: Any = None

    def load(self) -> None:
        model_path: Path = self.request["diarizationModelPath"]
        if not model_path.exists():
            raise WorkerError(
                "diarization_setup_required",
                f"The gated local Community-1 model is not installed at {model_path}.",
                exit_code=24,
                details={
                    "expectedPath": str(model_path),
                    "modelId": DIARIZATION_MODEL_ID,
                    "action": (
                        "Accept the Community-1 Hugging Face terms and run ADsum local "
                        "speaker-model setup."
                    ),
                },
            )
        try:
            import numpy as np
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise WorkerError(
                "runtime_missing",
                f"The local speech Python runtime is incomplete: {exc}",
                exit_code=20,
                details={"package": "pyannote.audio"},
            ) from exc
        device_name = str(self.request["device"])
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise WorkerError(
                "cuda_unavailable",
                "CUDA is unavailable for local speaker diarization.",
                retryable=True,
                exit_code=22,
            )
        checkpoint_path = model_path
        if model_path.is_dir():
            for filename in ("config.yaml", "config.yml"):
                candidate = model_path / filename
                if candidate.is_file():
                    checkpoint_path = candidate
                    break
            else:
                raise WorkerError(
                    "diarization_setup_required",
                    f"The local Community-1 snapshot has no config.yaml at {model_path}.",
                    exit_code=24,
                    details={"expectedPath": str(model_path / "config.yaml")},
                )
        try:
            pipeline = Pipeline.from_pretrained(str(checkpoint_path))
            pipeline.to(torch.device(device_name))
        except Exception as exc:
            if _is_cuda_out_of_memory(exc):
                raise WorkerError(
                    "cuda_out_of_memory",
                    "Community-1 could not fit in GPU memory after ASR was unloaded.",
                    retryable=True,
                    exit_code=23,
                ) from exc
            raise WorkerError(
                "diarization_model_load_failed",
                f"Could not load the local Community-1 model: {exc}",
                exit_code=24,
            ) from exc
        self.pipeline = pipeline
        self.torch = torch
        self.numpy = np

    def diarize(self) -> dict[str, Any]:
        if self.pipeline is None or self.torch is None or self.numpy is None:
            raise WorkerError("internal_error", "The diarization engine was not loaded.", exit_code=50)
        audio_load_started = time.monotonic()
        audio = _load_pcm_for_pyannote(
            self.request["audioPath"],
            self.torch,
            self.numpy,
        )
        audio_load_seconds = time.monotonic() - audio_load_started
        options: dict[str, int] = {}
        if self.request.get("numSpeakers") is not None:
            options["num_speakers"] = int(self.request["numSpeakers"])
        if self.request.get("minSpeakers") is not None:
            options["min_speakers"] = int(self.request["minSpeakers"])
        if self.request.get("maxSpeakers") is not None:
            options["max_speakers"] = int(self.request["maxSpeakers"])
        try:
            output = self.pipeline(audio, **options)
        except Exception as exc:
            if _is_cuda_out_of_memory(exc):
                raise WorkerError(
                    "cuda_out_of_memory",
                    "Community-1 ran out of GPU memory while diarizing the complete meeting.",
                    retryable=True,
                    exit_code=23,
                ) from exc
            raise WorkerError(
                "diarization_failed",
                f"Whole-meeting speaker diarization failed: {exc}",
                retryable=True,
            ) from exc
        exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
        regular_annotation = getattr(output, "speaker_diarization", None)
        if exclusive_annotation is None:
            raise WorkerError(
                "diarization_output_incompatible",
                "Community-1 did not provide exclusive_speaker_diarization for word assignment.",
                exit_code=24,
            )
        exclusive_turns = _extract_annotation_turns(exclusive_annotation)
        regular_turns = _extract_annotation_turns(
            exclusive_annotation if regular_annotation is None else regular_annotation
        )
        return {
            "exclusiveTurns": exclusive_turns,
            "regularTurns": regular_turns,
            "backend": "pyannote.audio",
            "audioLoadSeconds": audio_load_seconds,
        }

    def unload(self) -> None:
        self.pipeline = None
        self.torch = None
        self.numpy = None
        _cleanup_accelerator()


def _normalize_raw_turns(values: Iterable[Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for value in values:
        start_value = _read_attr(value, "start")
        end_value = _read_attr(value, "end")
        speaker_value = _read_attr(value, "speaker")
        if start_value is None or end_value is None or speaker_value is None:
            continue
        start = _finite_float(start_value, name="turn.start")
        end = _finite_float(end_value, name="turn.end")
        speaker = str(speaker_value).strip()
        if start >= 0 and end > start and speaker:
            turns.append({"start": start, "end": end, "speaker": speaker})
    turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return turns


def _canonicalize_turns(
    exclusive_raw: Sequence[Mapping[str, Any]],
    regular_raw: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    label_map: dict[str, str] = {}
    ordered = sorted(
        [*exclusive_raw, *regular_raw],
        key=lambda item: (float(item["start"]), float(item["end"]), str(item["speaker"])),
    )
    for turn in ordered:
        raw = str(turn["speaker"])
        if raw not in label_map:
            label_map[raw] = f"S{len(label_map) + 1:02d}"

    def convert(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "start": round(float(value["start"]), 6),
                "end": round(float(value["end"]), 6),
                "speaker": label_map[str(value["speaker"])],
            }
            for value in values
        ]

    return convert(exclusive_raw), convert(regular_raw), label_map


def _find_overlap_turns(regular_turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: dict[float, list[tuple[str, int]]] = {}
    for turn in regular_turns:
        start = float(turn["start"])
        end = float(turn["end"])
        speaker = str(turn["speaker"])
        events.setdefault(start, []).append((speaker, 1))
        events.setdefault(end, []).append((speaker, -1))
    active: dict[str, int] = {}
    previous: float | None = None
    overlaps: list[dict[str, Any]] = []
    for boundary in sorted(events):
        speakers = sorted(speaker for speaker, count in active.items() if count > 0)
        if previous is not None and boundary > previous and len(speakers) >= 2:
            if (
                overlaps
                and overlaps[-1]["speakers"] == speakers
                and math.isclose(float(overlaps[-1]["end"]), previous, abs_tol=1e-6)
            ):
                overlaps[-1]["end"] = round(boundary, 6)
            else:
                overlaps.append(
                    {
                        "start": round(previous, 6),
                        "end": round(boundary, 6),
                        "speakers": speakers,
                    }
                )
        # Applying all changes at the same boundary after recording the prior
        # interval avoids inventing zero-length overlaps at speaker handoffs.
        for speaker, delta in events[boundary]:
            active[speaker] = active.get(speaker, 0) + delta
            if active[speaker] <= 0:
                active.pop(speaker, None)
        previous = boundary
    return overlaps


def _speaker_for_interval(
    start: float,
    end: float,
    turns: Sequence[Mapping[str, Any]],
    turn_starts: Sequence[float],
) -> str | None:
    if not turns:
        return None
    position = bisect.bisect_right(turn_starts, start)
    candidate_indices = range(max(0, position - 2), min(len(turns), position + 3))
    best_speaker: str | None = None
    best_overlap = -1.0
    best_distance = math.inf
    midpoint = (start + end) / 2.0
    for index in candidate_indices:
        turn = turns[index]
        turn_start = float(turn["start"])
        turn_end = float(turn["end"])
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if turn_start <= midpoint <= turn_end:
            distance = 0.0
        else:
            distance = min(abs(midpoint - turn_start), abs(midpoint - turn_end))
        if overlap > best_overlap or (
            math.isclose(overlap, best_overlap, abs_tol=1e-9) and distance < best_distance
        ):
            best_speaker = str(turn["speaker"])
            best_overlap = overlap
            best_distance = distance
    return best_speaker


def _join_word_text(words: Sequence[Mapping[str, Any]]) -> str:
    output = ""
    for word in words:
        token = str(word["text"])
        if not output:
            output = token.lstrip()
        elif token[:1].isspace() or token[:1] in ".,!?;:%)]}":
            output += token
        else:
            output += " " + token
    return output.strip()


def assign_speakers_to_words(
    asr_segments: Sequence[Mapping[str, Any]],
    exclusive_turns: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign each ASR word from exclusive diarization and rebuild turns."""

    turns = sorted(
        exclusive_turns,
        key=lambda item: (float(item["start"]), float(item["end"])),
    )
    starts = [float(turn["start"]) for turn in turns]
    output_segments: list[dict[str, Any]] = []
    output_words: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for source_index, segment in enumerate(asr_segments):
        raw_words = list(segment.get("words") or [])
        assigned_words: list[dict[str, Any]] = []
        for word in raw_words:
            start = float(word["start"])
            end = float(word["end"])
            speaker = _speaker_for_interval(start, end, turns, starts)
            if speaker is None:
                raise WorkerError(
                    "diarization_empty",
                    "Community-1 returned no speaker turn for transcribed words.",
                    retryable=True,
                )
            assigned = {**word, "speaker": speaker}
            assigned_words.append(assigned)
            output_words.append(assigned)

        if not assigned_words:
            start = float(segment["start"])
            end = float(segment["end"])
            speaker = _speaker_for_interval(start, end, turns, starts)
            if speaker is None:
                raise WorkerError(
                    "diarization_empty",
                    "Community-1 returned no speaker turn for a transcribed segment.",
                    retryable=True,
                )
            warnings.append(
                {
                    "code": "missing_word_timestamps",
                    "segmentIndex": source_index,
                    "message": "A segment had no word timestamps; its midpoint speaker was used.",
                }
            )
            output_segments.append(
                {
                    "id": len(output_segments),
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "speaker": speaker,
                    "text": str(segment["text"]),
                    "words": [],
                }
            )
            continue

        group: list[dict[str, Any]] = []
        current_speaker: str | None = None
        for word in assigned_words:
            speaker = str(word["speaker"])
            if group and speaker != current_speaker:
                output_segments.append(
                    {
                        "id": len(output_segments),
                        "start": group[0]["start"],
                        "end": group[-1]["end"],
                        "speaker": current_speaker,
                        "text": _join_word_text(group),
                        "words": group,
                    }
                )
                group = []
            current_speaker = speaker
            group.append(word)
        if group:
            output_segments.append(
                {
                    "id": len(output_segments),
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "speaker": current_speaker,
                    "text": _join_word_text(group),
                    "words": group,
                }
            )

    return output_segments, output_words, warnings


def canonical_text(segments: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for segment in segments:
        parts.append(
            f"[{float(segment['start']):.2f}]"
            f"[{segment['speaker']}]"
            f"{str(segment['text']).strip()}"
            f"[{float(segment['end']):.2f}]"
        )
    return "".join(parts)


def _asr_signature(request: Mapping[str, Any], audio_sha256: str) -> str:
    payload = {
        "workerRevision": WORKER_REVISION,
        "audioSha256": audio_sha256,
        "modelId": ASR_MODEL_ID,
        "modelPath": str(request["asrModelPath"]),
        "language": request["language"],
        "hotwords": request["hotwords"],
        "device": request["device"],
        "computeType": request["computeType"],
        "vadFilter": request["vadFilter"],
        "wordTimestamps": True,
        "mockInference": bool(request["mockEnabled"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_directory(request: Mapping[str, Any], audio_sha256: str) -> Path:
    configured = request.get("checkpointDirectory")
    if configured is not None:
        return Path(configured)
    output_path: Path = request["outputPath"]
    return output_path.parent / ".local-speech-checkpoints" / audio_sha256[:16]


def _load_asr_checkpoint(
    path: Path,
    signature: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schemaVersion") != CHECKPOINT_SCHEMA_VERSION:
        return None
    if payload.get("jobSignature") != signature:
        return None
    asr = payload.get("asr")
    if not isinstance(asr, dict) or not isinstance(asr.get("segments"), list):
        return None
    return asr


def _preflight_model_paths(request: Mapping[str, Any], *, require_asr: bool) -> None:
    if require_asr and not Path(request["asrModelPath"]).is_dir():
        path = Path(request["asrModelPath"])
        raise WorkerError(
            "asr_setup_required",
            f"The local faster-whisper large-v3-turbo model is not installed at {path}.",
            exit_code=21,
            details={"expectedPath": str(path), "modelId": ASR_MODEL_ID},
        )
    diarization_path = Path(request["diarizationModelPath"])
    if not diarization_path.exists():
        raise WorkerError(
            "diarization_setup_required",
            f"The gated local Community-1 model is not installed at {diarization_path}.",
            exit_code=24,
            details={
                "expectedPath": str(diarization_path),
                "modelId": DIARIZATION_MODEL_ID,
                "action": "Accept the model terms and run ADsum speaker-model setup.",
            },
        )
    if diarization_path.is_dir() and not any(
        (diarization_path / filename).is_file()
        for filename in ("config.yaml", "config.yml")
    ):
        raise WorkerError(
            "diarization_setup_required",
            f"The local Community-1 snapshot has no config.yaml at {diarization_path}.",
            exit_code=24,
            details={"expectedPath": str(diarization_path / "config.yaml")},
        )


def run_request(raw_request: Mapping[str, Any], emit: EventSink = emit_ndjson) -> dict[str, Any]:
    job_started = time.monotonic()
    request = normalize_request(raw_request)
    request_id = str(request["requestId"])
    reporter = ProgressReporter(
        request_id,
        emit,
        job_started,
    )
    emit(
        {
            "type": "started",
            "requestId": request_id,
            "modelRevision": "large-v3-turbo",
            "postRecording": True,
        }
    )

    inspect_started = time.monotonic()
    reporter.progress("inspecting_audio", 0.01, stage_started=inspect_started)
    wav_info = inspect_wav(request["audioPath"])
    audio_sha256 = sha256_file(request["audioPath"])
    inspect_seconds = time.monotonic() - inspect_started

    checkpoint_directory = _checkpoint_directory(request, audio_sha256)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_directory / "asr-checkpoint.json"
    signature = _asr_signature(request, audio_sha256)
    checkpoint_asr = (
        _load_asr_checkpoint(checkpoint_path, signature) if request["resume"] else None
    )

    if not request["mockEnabled"]:
        _preflight_model_paths(request, require_asr=checkpoint_asr is None)

    emit(
        {
            "type": "chunk_started",
            "requestId": request_id,
            "index": 0,
            "total": 1,
            "start": 0.0,
            "end": round(wav_info.duration_seconds, 6),
            "strategy": "full_file",
        }
    )

    warnings: list[Any] = []
    asr_load_seconds = 0.0
    asr_seconds = 0.0
    asr_unload_seconds = 0.0
    asr_resumed = checkpoint_asr is not None

    if checkpoint_asr is not None:
        asr_result = checkpoint_asr
        selected_batch_size = int(asr_result.get("selectedBatchSize", request["batchSize"]))
        batch_attempts = list(asr_result.get("batchAttempts") or [])
        source_asr_seconds = float(asr_result.get("sourceAsrSeconds", 0.0))
        reporter.progress(
            "asr_checkpoint_resumed",
            0.58,
            checkpointPath=str(checkpoint_path),
            selectedBatchSize=selected_batch_size,
        )
    else:
        asr_engine: MockAsrEngine | FasterWhisperEngine
        asr_engine = (
            MockAsrEngine(request, wav_info)
            if request["mockEnabled"]
            else FasterWhisperEngine(request)
        )
        load_started = time.monotonic()
        reporter.progress("loading_asr_model", 0.05, stage_started=load_started)
        asr_engine.load()
        asr_load_seconds = time.monotonic() - load_started
        asr_started = time.monotonic()
        try:
            asr_result, batch_attempts, selected_batch_size = _run_asr_with_fallback(
                asr_engine,
                request,
                reporter,
            )
            asr_seconds = time.monotonic() - asr_started
            source_asr_seconds = asr_load_seconds + asr_seconds
        finally:
            unload_started = time.monotonic()
            reporter.progress("releasing_asr_model", 0.56, stage_started=unload_started)
            asr_engine.unload()
            asr_unload_seconds = time.monotonic() - unload_started

        asr_result = {
            **asr_result,
            "selectedBatchSize": selected_batch_size,
            "batchAttempts": batch_attempts,
            "sourceAsrSeconds": round(source_asr_seconds, 6),
        }
        checkpoint_payload = {
            "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
            "jobSignature": signature,
            "audioSha256": audio_sha256,
            "model": {
                "id": ASR_MODEL_ID,
                "path": str(request["asrModelPath"]),
            },
            "asr": asr_result,
        }
        _atomic_write_json(checkpoint_path, checkpoint_payload)
        reporter.progress(
            "asr_checkpoint_saved",
            0.58,
            checkpointPath=str(checkpoint_path),
            selectedBatchSize=selected_batch_size,
        )

    reporter.progress(
        "asr_completed",
        0.60,
        selectedBatchSize=selected_batch_size,
        segmentCount=len(asr_result.get("segments") or []),
        resumed=asr_resumed,
    )

    diarization_engine: MockDiarizationEngine | PyannoteDiarizationEngine
    diarization_engine = (
        MockDiarizationEngine(request, wav_info)
        if request["mockEnabled"]
        else PyannoteDiarizationEngine(request)
    )
    diarization_load_started = time.monotonic()
    reporter.progress(
        "loading_diarization_model",
        0.62,
        stage_started=diarization_load_started,
    )
    diarization_engine.load()
    diarization_load_seconds = time.monotonic() - diarization_load_started

    diarization_started = time.monotonic()
    reporter.progress(
        "diarizing_audio",
        0.68,
        stage_started=diarization_started,
    )
    try:
        diarization_raw = diarization_engine.diarize()
        diarization_seconds = time.monotonic() - diarization_started
    finally:
        diarization_engine.unload()

    merge_started = time.monotonic()
    reporter.progress("merging_speakers", 0.90, stage_started=merge_started)
    exclusive_raw = _normalize_raw_turns(diarization_raw.get("exclusiveTurns") or [])
    regular_raw = _normalize_raw_turns(diarization_raw.get("regularTurns") or [])
    exclusive_turns, regular_turns, raw_label_map = _canonicalize_turns(
        exclusive_raw,
        regular_raw,
    )
    asr_segments = list(asr_result.get("segments") or [])
    if asr_segments and not exclusive_turns:
        raise WorkerError(
            "diarization_empty",
            "Community-1 returned no speaker turns for a non-empty transcription.",
            retryable=True,
        )
    if asr_segments:
        segments, words, assignment_warnings = assign_speakers_to_words(
            asr_segments,
            exclusive_turns,
        )
    else:
        segments, words, assignment_warnings = [], [], []
        warnings.append("No speech was detected in the completed recording.")
    warnings.extend(assignment_warnings)
    overlap_turns = _find_overlap_turns(regular_turns)
    merge_seconds = time.monotonic() - merge_started

    reporter.progress("writing_result", 0.97, stage_started=time.monotonic())

    chunk_summary = {
        "index": 0,
        "start": 0.0,
        "end": round(wav_info.duration_seconds, 6),
        "duration": round(wav_info.duration_seconds, 6),
        "strategy": "full_file",
        "segmentCount": len(segments),
        "resumed": asr_resumed,
    }

    result: dict[str, Any] = {
        "schemaVersion": RESULT_SCHEMA_VERSION,
        "requestId": request_id,
        "model": {
            "id": ASR_MODEL_ID,
            "revision": "large-v3-turbo",
            "path": str(request["asrModelPath"]),
            "backend": str(asr_result.get("backend", "unknown")),
            "diarization": {
                "id": DIARIZATION_MODEL_ID,
                "path": str(request["diarizationModelPath"]),
                "backend": str(diarization_raw.get("backend", "unknown")),
            },
        },
        "audio": {
            "path": str(request["audioPath"]),
            "durationSeconds": round(wav_info.duration_seconds, 6),
            "sha256": audio_sha256,
            "sampleRate": wav_info.sample_rate,
            "channels": wav_info.channels,
        },
        "language": {
            "requested": request["language"],
            "detected": asr_result.get("detectedLanguage"),
            "probability": asr_result.get("languageProbability"),
        },
        "text": canonical_text(segments),
        "segments": segments,
        "words": words,
        "chunks": [chunk_summary],
        "coverage": {
            "complete": True,
            "coveredUntil": round(wav_info.duration_seconds, 6),
            "audioDuration": round(wav_info.duration_seconds, 6),
            "transcriptionLastEnd": round(
                max((float(segment["end"]) for segment in segments), default=0.0),
                6,
            ),
        },
        "diarization": {
            "exclusiveTurns": exclusive_turns,
            "regularTurns": regular_turns,
            "overlapTurns": overlap_turns,
            "speakerCount": len(set(raw_label_map.values())),
            "rawLabelMap": raw_label_map,
            "wholeMeeting": True,
        },
        "timings": {
            "inspectSeconds": round(inspect_seconds, 6),
            "loadSeconds": round(asr_load_seconds + diarization_load_seconds, 6),
            "asrLoadSeconds": round(asr_load_seconds, 6),
            "asrSeconds": round(asr_seconds, 6),
            "asrUnloadSeconds": round(asr_unload_seconds, 6),
            "sourceAsrSeconds": round(source_asr_seconds, 6),
            "diarizationLoadSeconds": round(diarization_load_seconds, 6),
            "diarizationAudioLoadSeconds": round(
                float(diarization_raw.get("audioLoadSeconds", 0.0)),
                6,
            ),
            "diarizationSeconds": round(diarization_seconds, 6),
            "mergeSeconds": round(merge_seconds, 6),
            # Replaced with the final wall time immediately before the atomic write.
            "totalSeconds": 0.0,
        },
        "performance": {
            "selectedBatchSize": selected_batch_size,
            "batchAttempts": batch_attempts,
            "asrResumed": asr_resumed,
            "audioRealtimeFactor": 0.0,
        },
        "warnings": warnings,
        "checkpointDirectory": str(checkpoint_directory),
        "asrCheckpointPath": str(checkpoint_path),
    }

    total_seconds = time.monotonic() - job_started
    result["timings"]["totalSeconds"] = round(total_seconds, 6)
    result["performance"]["audioRealtimeFactor"] = round(
        total_seconds / wav_info.duration_seconds,
        6,
    )
    _atomic_write_json(request["outputPath"], result)

    emit(
        {
            "type": "chunk_completed",
            "requestId": request_id,
            "index": 0,
            "total": 1,
            "resumed": asr_resumed,
            "segmentCount": len(segments),
            "progress": 0.99,
            "elapsedSeconds": round(time.monotonic() - job_started, 6),
        }
    )
    reporter.progress("completed", 1.0, eta_seconds=0.0)
    emit(
        {
            "type": "completed",
            "requestId": request_id,
            "resultPath": str(request["outputPath"]),
            "segmentCount": len(segments),
            "wordCount": len(words),
            "speakerCount": len(set(raw_label_map.values())),
            "warningCount": len(warnings),
            "warnings": warnings,
            "elapsedSeconds": round(time.monotonic() - job_started, 6),
        }
    )
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one post-recording local ASR and speaker-diarization job."
    )
    parser.add_argument(
        "--request-file",
        help="Path to a UTF-8 JSON request. Reads stdin when omitted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    request_id: str | None = None
    try:
        raw_request = load_request(args.request_file)
        request_id = str(raw_request.get("requestId") or "") or None
        run_request(raw_request)
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
                "message": "The local post-recording transcription job was cancelled.",
                "retryable": True,
            }
        )
        return 40
    except Exception as exc:  # pragma: no cover - final process safety boundary
        print(f"Unexpected local speech worker failure: {exc!r}", file=sys.stderr, flush=True)
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
