from __future__ import annotations

import importlib.util
import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "src" / "ADsum.Desktop" / "Moss" / "moss_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("adsum_moss_worker_tests", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_worker()


def _write_wav(path: Path, duration: float, *, amplitude: int = 1_000) -> None:
    sample_rate = 16_000
    frames = int(round(duration * sample_rate))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        block = bytearray()
        for index in range(frames):
            sample = int(amplitude * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
            block.extend(struct.pack("<h", sample))
            if len(block) >= sample_rate * 2:
                writer.writeframesraw(block)
                block.clear()
        if block:
            writer.writeframesraw(block)
        writer.writeframes(b"")


def test_parse_canonical_transcript() -> None:
    raw = (
        "[0.48][S1]Welcome everyone[1.66]"
        "[12.26][S02]The pipeline is ready[13.81]"
    )

    segments = worker.parse_canonical_transcript(raw)

    assert segments == [
        {"start": 0.48, "end": 1.66, "speaker": "S01", "text": "Welcome everyone"},
        {"start": 12.26, "end": 13.81, "speaker": "S02", "text": "The pipeline is ready"},
    ]
    assert worker.parse_canonical_transcript(worker.canonical_text(segments)) == segments


def test_long_audio_plan_uses_five_minutes_with_overlap() -> None:
    chunks = worker.build_chunk_plan(2 * 60 * 60)

    assert chunks[0].start == 0
    assert chunks[-1].end == 2 * 60 * 60
    assert len(chunks) == 27
    assert all(chunk.duration <= 5 * 60 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start == pytest.approx(previous.end - 30)


def test_extract_wav_chunk_streams_exact_interval(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    destination = tmp_path / "chunk.wav"
    _write_wav(source, 4.0)
    chunk = worker.ChunkSpec(index=0, start=1.25, end=2.75)

    worker.extract_wav_chunk(source, destination, chunk)

    info = worker.inspect_wav(destination)
    assert info.duration_seconds == pytest.approx(1.5, abs=1 / 16_000)
    assert destination.stat().st_size < source.stat().st_size


def test_cache_mode_defaults_to_fast_auto_with_legacy_compatibility(tmp_path: Path) -> None:
    audio = tmp_path / "cache-mode.wav"
    _write_wav(audio, 1.0)
    base = {
        "audioPath": str(audio),
        "outputPath": str(tmp_path / "result.json"),
    }

    assert worker.normalize_request(base)["cacheMode"] == "auto"
    assert worker.normalize_request({**base, "cacheMode": "dynamic"})["cacheMode"] == "gpu"
    assert worker.normalize_request({**base, "offloadCache": True})["cacheMode"] == "offloaded"
    assert worker.normalize_request({**base, "offloadCache": False})["cacheMode"] == "gpu"


def test_invalid_cache_mode_is_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "invalid-cache-mode.wav"
    _write_wav(audio, 1.0)

    with pytest.raises(worker.WorkerError) as error:
        worker.normalize_request(
            {
                "audioPath": str(audio),
                "outputPath": str(tmp_path / "result.json"),
                "cacheMode": "mystery",
            }
        )

    assert error.value.code == "invalid_cache_mode"


def test_auto_cache_oom_retries_offloaded_and_keeps_using_it(monkeypatch) -> None:
    engine = object.__new__(worker.MossInferenceEngine)
    engine.request = {"cacheMode": "auto"}
    engine._force_offloaded_cache = False
    events: list[dict[str, object]] = []
    engine.emit = events.append
    calls: list[str] = []
    cleanup_calls: list[str] = []

    def generate_once(_inputs, _max_new_tokens: int, cache_mode: str):
        calls.append(cache_mode)
        if calls == ["gpu"]:
            raise RuntimeError("CUDA out of memory while allocating the dynamic cache")
        return f"generated-with-{cache_mode}"

    class FakeCuda:
        @staticmethod
        def empty_cache() -> None:
            cleanup_calls.append("empty_cache")

    class FakeTorch:
        cuda = FakeCuda()

    engine._generate_once = generate_once
    engine._torch = FakeTorch()
    monkeypatch.setattr(worker.gc, "collect", lambda: cleanup_calls.append("gc"))
    chunk = worker.ChunkSpec(index=3, start=810.0, end=1110.0)

    first = engine._generate_with_cache_fallback({}, 4096, chunk)
    second = engine._generate_with_cache_fallback({}, 4096, chunk)

    assert first == ("generated-with-offloaded", "offloaded", True)
    assert second == ("generated-with-offloaded", "offloaded", False)
    assert calls == ["gpu", "offloaded", "offloaded"]
    assert cleanup_calls == ["gc", "empty_cache"]
    assert engine._force_offloaded_cache is True
    assert events == [
        {
            "type": "progress",
            "phase": "cache_fallback_offloaded",
            "chunkIndex": 3,
        }
    ]


def test_forced_gpu_cache_does_not_hide_out_of_memory() -> None:
    engine = object.__new__(worker.MossInferenceEngine)
    engine.request = {"cacheMode": "gpu"}
    engine._force_offloaded_cache = False
    engine.emit = lambda _event: None
    engine._generate_once = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory in forced GPU mode")
    )
    chunk = worker.ChunkSpec(index=0, start=0.0, end=300.0)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        engine._generate_with_cache_fallback({}, 4096, chunk)


def test_effective_silence_peak_is_deliberately_tiny(tmp_path: Path) -> None:
    digital_silence = tmp_path / "digital-silence.wav"
    quantization_residue = tmp_path / "quantization-residue.wav"
    audible_signal = tmp_path / "audible-signal.wav"
    _write_wav(digital_silence, 0.25, amplitude=0)
    _write_wav(quantization_residue, 0.25, amplitude=worker.EFFECTIVE_SILENCE_PEAK)
    _write_wav(audible_signal, 0.25, amplitude=worker.EFFECTIVE_SILENCE_PEAK + 2)

    assert worker.wav_is_effectively_silent(digital_silence) is True
    assert worker.wav_is_effectively_silent(quantization_residue) is True
    assert worker.wav_is_effectively_silent(audible_signal) is False


def test_silent_request_succeeds_empty_without_constructing_model(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "silent-recording.wav"
    result_path = tmp_path / "silent-recording.moss.json"
    _write_wav(audio, 1.0, amplitude=0)

    class ModelMustNotBeConstructed:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("MOSS engine was constructed for silent audio")

    monkeypatch.setattr(worker, "MossInferenceEngine", ModelMustNotBeConstructed)
    events: list[dict] = []
    result = worker.run_request(
        {
            "requestId": "silent-job",
            "audioPath": str(audio),
            "outputPath": str(result_path),
        },
        events.append,
    )

    assert result["text"] == ""
    assert result["segments"] == []
    assert result["chunks"] == []
    assert result["coverage"] == {
        "complete": True,
        "coveredUntil": 1.0,
        "audioDuration": 1.0,
    }
    assert result_path.is_file()
    assert any(event.get("phase") == "silence_detected" for event in events)
    assert events[-1]["type"] == "completed"
    assert events[-1]["segmentCount"] == 0


def test_non_silent_malformed_output_still_fails_validation(tmp_path: Path) -> None:
    audio = tmp_path / "quiet-but-real-signal.wav"
    result_path = tmp_path / "quiet-but-real-signal.moss.json"
    _write_wav(audio, 1.0, amplitude=worker.EFFECTIVE_SILENCE_PEAK + 2)

    with pytest.raises(worker.WorkerError) as captured:
        worker.run_request(
            {
                "requestId": "malformed-job",
                "audioPath": str(audio),
                "outputPath": str(result_path),
                "mockInference": {
                    "chunks": [{"rawText": "This is not the canonical MOSS format."}]
                },
            },
            lambda _event: None,
        )

    assert captured.value.code == "malformed_transcript"
    assert captured.value.retryable is True
    assert not result_path.exists()


def test_validate_rejects_speech_after_last_timestamp(tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    _write_wav(audio, 5.0)
    chunk = worker.ChunkSpec(index=3, start=0.0, end=5.0)
    segments = [{"start": 0.0, "end": 1.0, "speaker": "S01", "text": "Too short"}]

    with pytest.raises(worker.WorkerError) as captured:
        worker.validate_chunk_result(
            audio,
            chunk,
            worker.canonical_text(segments),
            segments,
            coverage_slack_seconds=0.1,
        )

    assert captured.value.code == "incomplete_transcript"
    assert captured.value.retryable is True


def test_validate_rejects_pathological_repetition(tmp_path: Path) -> None:
    audio = tmp_path / "silence.wav"
    _write_wav(audio, 4.0, amplitude=0)
    chunk = worker.ChunkSpec(index=0, start=0.0, end=4.0)
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "S01", "text": "please repeat this phrase"},
        {"start": 1.0, "end": 2.0, "speaker": "S01", "text": "please repeat this phrase"},
        {"start": 2.0, "end": 3.0, "speaker": "S01", "text": "please repeat this phrase"},
    ]

    with pytest.raises(worker.WorkerError) as captured:
        worker.validate_chunk_result(audio, chunk, worker.canonical_text(segments), segments)

    assert captured.value.code == "repeated_generation"


def test_overlap_maps_speaker_and_removes_duplicate_turn() -> None:
    chunk_results = [
        {
            "index": 0,
            "start": 0.0,
            "end": 10.0,
            "segments": [
                {"start": 0.0, "end": 1.0, "speaker": "S01", "text": "Opening"},
                {"start": 8.2, "end": 9.8, "speaker": "S02", "text": "Shared turn"},
            ],
        },
        {
            "index": 1,
            "start": 8.0,
            "end": 18.0,
            "segments": [
                {"start": 0.2, "end": 1.8, "speaker": "S07", "text": "Shared turn"},
                {"start": 2.4, "end": 3.4, "speaker": "S08", "text": "A new person"},
            ],
        },
    ]

    segments, warnings, mappings = worker.merge_chunk_results(chunk_results)

    assert [segment["text"] for segment in segments].count("Shared turn") == 1
    shared = next(segment for segment in segments if segment["text"] == "Shared turn")
    assert shared["speaker"] == "S02"
    assert mappings[1]["S07"] == "S02"
    assert any("local S08" in warning for warning in warnings)


def test_mock_cli_writes_result_and_resumes_checkpoints(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    result_path = tmp_path / "meeting.moss.json"
    checkpoint_dir = tmp_path / "checkpoints"
    request_path = tmp_path / "request.json"
    _write_wav(audio, 8.0)
    request = {
        "protocolVersion": 1,
        "requestId": "mock-job",
        "audioPath": str(audio),
        "outputPath": str(result_path),
        "checkpointDirectory": str(checkpoint_dir),
        "chunkSeconds": 5,
        "overlapSeconds": 2,
        "coverageSlackSeconds": 0.01,
        "mockInference": {
            "chunks": [
                {
                    "segments": [
                        {"start": 0.0, "end": 1.0, "speaker": "S01", "text": "Opening"},
                        {"start": 3.2, "end": 4.99, "speaker": "S02", "text": "Shared turn"},
                    ]
                },
                {
                    "segments": [
                        {"start": 0.2, "end": 1.99, "speaker": "S07", "text": "Shared turn"},
                        {"start": 2.2, "end": 4.99, "speaker": "S08", "text": "New speaker"},
                    ]
                },
            ]
        },
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")

    first = subprocess.run(
        [sys.executable, str(WORKER_PATH), "--request-file", str(request_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr + first.stdout
    events = [json.loads(line) for line in first.stdout.splitlines() if line.strip()]
    assert events[0]["type"] == "started"
    assert events[-1]["type"] == "completed"
    assert result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["model"]["revision"] == worker.MODEL_REVISION
    assert result["coverage"]["complete"] is True
    assert len(result["chunks"]) == 2
    assert result["performance"]["cacheModes"] == ["auto"]
    assert result["performance"]["cacheFallbackCount"] == 0
    assert result["performance"]["totalGeneratedTokens"] > 0
    assert [segment["text"] for segment in result["segments"]].count("Shared turn") == 1
    assert next(segment for segment in result["segments"] if segment["text"] == "Shared turn")["speaker"] == "S02"

    second = subprocess.run(
        [sys.executable, str(WORKER_PATH), "--request-file", str(request_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert second.returncode == 0, second.stderr + second.stdout
    resumed_events = [json.loads(line) for line in second.stdout.splitlines() if line.strip()]
    completed_chunks = [event for event in resumed_events if event["type"] == "chunk_completed"]
    assert completed_chunks
    assert all(event["resumed"] is True for event in completed_chunks)
    resumed_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert resumed_result["performance"]["resumedChunkCount"] == 2
    assert all(chunk["resumed"] is True for chunk in resumed_result["chunks"])
    assert (
        resumed_result["performance"]["totalInferenceSeconds"]
        == result["performance"]["totalInferenceSeconds"]
    )


def test_import_does_not_load_torch() -> None:
    script = (
        "import importlib.util,sys;"
        f"p={str(WORKER_PATH)!r};"
        "s=importlib.util.spec_from_file_location('moss_lazy_check',p);"
        "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
        "print('torch' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"
