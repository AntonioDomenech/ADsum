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
WORKER_PATH = ROOT / "src" / "ADsum.Desktop" / "Moss" / "local_speech_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("adsum_local_speech_worker_tests", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_worker()


def _write_wav(path: Path, duration: float, *, frequency: float = 220.0) -> None:
    sample_rate = 16_000
    frame_count = int(round(duration * sample_rate))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        block = bytearray()
        for index in range(frame_count):
            sample = int(1_000 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
            block.extend(struct.pack("<h", sample))
            if len(block) >= sample_rate * 2:
                writer.writeframesraw(block)
                block.clear()
        if block:
            writer.writeframesraw(block)
        writer.writeframes(b"")


def _speaker_reentry_request(
    audio: Path,
    output: Path,
    checkpoint_directory: Path,
) -> dict:
    return {
        "protocolVersion": 1,
        "requestId": "speaker-reentry",
        "audioPath": str(audio),
        "outputPath": str(output),
        "checkpointDirectory": str(checkpoint_directory),
        "language": "mixed",
        "hotwords": ["CERTANIA", "ADsum"],
        "batchSize": 8,
        "recordingComplete": True,
        "mockInference": {
            "detectedLanguage": "es",
            "asrSegments": [
                {
                    "start": 0.2,
                    "end": 1.2,
                    "text": "Alice opens",
                    "words": [
                        {"start": 0.2, "end": 0.6, "word": "Alice", "probability": 0.99},
                        {"start": 0.7, "end": 1.2, "word": " opens", "probability": 0.98},
                    ],
                },
                {
                    "start": 4.0,
                    "end": 4.8,
                    "text": "Bob answers",
                    "words": [
                        {"start": 4.0, "end": 4.3, "word": "Bob", "probability": 0.97},
                        {"start": 4.4, "end": 4.8, "word": " answers", "probability": 0.96},
                    ],
                },
                {
                    "start": 9.1,
                    "end": 10.2,
                    "text": "Alice returns",
                    "words": [
                        {"start": 9.1, "end": 9.5, "word": "Alice", "probability": 0.99},
                        {"start": 9.6, "end": 10.2, "word": " returns", "probability": 0.98},
                    ],
                },
            ],
            "exclusiveSpeakerTurns": [
                {"start": 0.0, "end": 2.0, "speaker": "alice-voice"},
                {"start": 2.0, "end": 8.0, "speaker": "bob-voice"},
                {"start": 8.0, "end": 12.0, "speaker": "alice-voice"},
            ],
            "regularSpeakerTurns": [
                {"start": 0.0, "end": 2.0, "speaker": "alice-voice"},
                {"start": 2.0, "end": 8.0, "speaker": "bob-voice"},
                {"start": 8.0, "end": 12.0, "speaker": "alice-voice"},
                {"start": 9.5, "end": 10.5, "speaker": "bob-voice"},
            ],
        },
    }


def test_normalize_request_is_post_recording_and_multilingual(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    _write_wav(audio, 1.0)
    normalized = worker.normalize_request(
        {
            "audioPath": str(audio),
            "outputPath": str(tmp_path / "result.json"),
            "language": "catalan",
            "mockInference": True,
        }
    )

    assert normalized["language"] == "ca"
    assert normalized["batchSize"] == 8
    assert normalized["wordTimestamps"] is True
    assert "hardDeadlineSeconds" not in normalized
    assert "optionalRefinementCutoffSeconds" not in normalized

    with pytest.raises(worker.WorkerError) as captured:
        worker.normalize_request(
            {
                "audioPath": str(audio),
                "outputPath": str(tmp_path / "never.json"),
                "recordingComplete": False,
            }
        )
    assert captured.value.code == "recording_not_complete"


def test_full_file_mock_assigns_words_and_preserves_returning_speaker(tmp_path: Path) -> None:
    audio = tmp_path / "long-meeting.wav"
    output = tmp_path / "result.json"
    checkpoints = tmp_path / "durable-checkpoints"
    _write_wav(audio, 12.0)
    events: list[dict] = []

    result = worker.run_request(
        _speaker_reentry_request(audio, output, checkpoints),
        events.append,
    )

    assert output.is_file()
    assert (checkpoints / "asr-checkpoint.json").is_file()
    assert [segment["speaker"] for segment in result["segments"]] == ["S01", "S02", "S01"]
    assert [segment["text"] for segment in result["segments"]] == [
        "Alice opens",
        "Bob answers",
        "Alice returns",
    ]
    assert all(word["speaker"] in {"S01", "S02"} for word in result["words"])
    assert len(result["words"]) == 6
    assert result["diarization"]["rawLabelMap"] == {
        "alice-voice": "S01",
        "bob-voice": "S02",
    }
    assert result["diarization"]["overlapTurns"] == [
        {"start": 9.5, "end": 10.5, "speakers": ["S01", "S02"]}
    ]
    assert result["diarization"]["wholeMeeting"] is True
    assert result["chunks"] == [
        {
            "index": 0,
            "start": 0.0,
            "end": 12.0,
            "duration": 12.0,
            "strategy": "full_file",
            "segmentCount": 3,
            "resumed": False,
        }
    ]
    assert result["coverage"]["complete"] is True
    assert result["coverage"]["coveredUntil"] == 12.0
    assert result["performance"]["selectedBatchSize"] == 8
    assert result["performance"]["asrResumed"] is False
    assert "deadline" not in result
    assert result["timings"]["totalSeconds"] >= 0
    assert result["timings"]["asrSeconds"] >= 0
    assert result["timings"]["diarizationSeconds"] >= 0
    assert result["timings"]["mergeSeconds"] >= 0

    assert events[0]["type"] == "started"
    assert events[0]["postRecording"] is True
    logical_chunks = [event for event in events if event["type"] == "chunk_started"]
    assert len(logical_chunks) == 1
    assert logical_chunks[0]["strategy"] == "full_file"
    progress_events = [event for event in events if event["type"] == "progress"]
    assert progress_events
    assert all("elapsedSeconds" in event for event in progress_events)
    assert all("deadlineRemainingSeconds" not in event for event in progress_events)
    assert events[-1]["type"] == "completed"
    assert events[-1]["speakerCount"] == 2


def test_cuda_oom_retries_batch_eight_then_four_then_two(tmp_path: Path) -> None:
    audio = tmp_path / "oom.wav"
    _write_wav(audio, 2.0)
    request = {
        "requestId": "oom-fallback",
        "audioPath": str(audio),
        "outputPath": str(tmp_path / "oom-result.json"),
        "checkpointDirectory": str(tmp_path / "oom-checkpoints"),
        "batchSize": 8,
        "mockInference": {
            "oomBatchSizes": [8, 4],
            "asrSegments": [
                {
                    "start": 0.1,
                    "end": 1.0,
                    "text": "Fallback succeeded",
                    "words": [
                        {"start": 0.1, "end": 0.5, "word": "Fallback"},
                        {"start": 0.6, "end": 1.0, "word": " succeeded"},
                    ],
                }
            ],
            "exclusiveSpeakerTurns": [
                {"start": 0.0, "end": 2.0, "speaker": "speaker-one"}
            ],
        },
    }
    events: list[dict] = []

    result = worker.run_request(request, events.append)

    assert result["performance"]["selectedBatchSize"] == 2
    assert [attempt["batchSize"] for attempt in result["performance"]["batchAttempts"]] == [
        8,
        4,
        2,
    ]
    assert [attempt["succeeded"] for attempt in result["performance"]["batchAttempts"]] == [
        False,
        False,
        True,
    ]
    fallback_events = [
        event
        for event in events
        if event.get("type") == "progress" and event.get("phase") == "batch_size_fallback"
    ]
    assert [(event["failedBatchSize"], event["nextBatchSize"]) for event in fallback_events] == [
        (8, 4),
        (4, 2),
    ]


def test_faster_whisper_uses_benchmarked_quality_options(tmp_path: Path) -> None:
    audio = tmp_path / "options.wav"
    _write_wav(audio, 1.0)
    request = worker.normalize_request(
        {
            "audioPath": str(audio),
            "outputPath": str(tmp_path / "options.json"),
            "language": "mixed",
            "hotwords": ["ADsum"],
            "mockInference": True,
        }
    )
    engine = worker.FasterWhisperEngine(request)
    calls: list[tuple[str, dict]] = []

    class FakePipeline:
        def transcribe(self, path: str, **options):
            calls.append((path, options))
            return iter(
                [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 0.5,
                        "text": "hello",
                        "words": [{"start": 0.0, "end": 0.5, "word": "hello"}],
                    }
                ]
            ), {"language": "en", "language_probability": 0.9}

    engine.pipeline = FakePipeline()
    result = engine.transcribe(8)

    assert result["detectedLanguage"] == "en"
    assert len(calls) == 1
    path, options = calls[0]
    assert path == str(audio.resolve())
    assert options == {
        "batch_size": 8,
        "language": None,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "word_timestamps": True,
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "without_timestamps": False,
        "multilingual": True,
        "hotwords": "ADsum",
    }


def test_pyannote_receives_preloaded_waveform_not_wav_filename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "preloaded.wav"
    _write_wav(audio, 1.0)
    request = worker.normalize_request(
        {
            "audioPath": str(audio),
            "outputPath": str(tmp_path / "preloaded.json"),
            "mockInference": True,
        }
    )
    engine = worker.PyannoteDiarizationEngine(request)
    waveform_mapping = {"waveform": object(), "sample_rate": 16_000}
    monkeypatch.setattr(
        worker,
        "_load_pcm_for_pyannote",
        lambda path, _torch, _numpy: waveform_mapping if path == audio.resolve() else None,
    )

    class FakeAnnotation:
        def itertracks(self, *, yield_label: bool):
            assert yield_label is True
            turn = type("Turn", (), {"start": 0.0, "end": 1.0})()
            yield turn, "track", "speaker"

    output = type(
        "Output",
        (),
        {
            "speaker_diarization": FakeAnnotation(),
            "exclusive_speaker_diarization": FakeAnnotation(),
        },
    )()
    received: list[object] = []

    class FakePipeline:
        def __call__(self, audio_input, **options):
            received.append(audio_input)
            assert options == {}
            return output

    engine.pipeline = FakePipeline()
    engine.torch = object()
    engine.numpy = object()
    result = engine.diarize()

    assert received == [waveform_mapping]
    assert not isinstance(received[0], (str, Path))
    assert result["exclusiveTurns"] == [
        {"start": 0.0, "end": 1.0, "speaker": "speaker"}
    ]


def test_asr_checkpoint_resumes_before_diarization(tmp_path: Path) -> None:
    audio = tmp_path / "resume.wav"
    output = tmp_path / "resume-result.json"
    checkpoints = tmp_path / "resume-checkpoints"
    _write_wav(audio, 12.0)
    request = _speaker_reentry_request(audio, output, checkpoints)
    first = worker.run_request(request, lambda _event: None)
    assert first["performance"]["asrResumed"] is False

    second_events: list[dict] = []
    second = worker.run_request(request, second_events.append)

    assert second["performance"]["asrResumed"] is True
    assert second["timings"]["asrSeconds"] == 0.0
    assert second["timings"]["sourceAsrSeconds"] == first["timings"]["sourceAsrSeconds"]
    assert any(event.get("phase") == "asr_checkpoint_resumed" for event in second_events)
    assert not any(event.get("phase") == "transcribing_audio" for event in second_events)
    assert [segment["speaker"] for segment in second["segments"]] == ["S01", "S02", "S01"]


def test_overwritten_wav_cannot_reuse_stale_asr_checkpoint(tmp_path: Path) -> None:
    audio = tmp_path / "overwritten.wav"
    output = tmp_path / "overwritten-result.json"
    checkpoints = tmp_path / "stable-path-checkpoints"
    _write_wav(audio, 12.0, frequency=220.0)
    request = _speaker_reentry_request(audio, output, checkpoints)
    first = worker.run_request(request, lambda _event: None)
    assert first["performance"]["asrResumed"] is False

    # Same path and duration, different PCM bytes: the content hash changes.
    _write_wav(audio, 12.0, frequency=440.0)
    second = worker.run_request(request, lambda _event: None)

    assert second["performance"]["asrResumed"] is False
    assert second["audio"]["sha256"] != first["audio"]["sha256"]


def test_missing_gated_model_fails_before_real_inference_import(tmp_path: Path) -> None:
    audio = tmp_path / "setup.wav"
    _write_wav(audio, 1.0)
    asr_model = tmp_path / "asr-model"
    asr_model.mkdir()
    missing_diarizer = tmp_path / "missing-community-1"

    with pytest.raises(worker.WorkerError) as captured:
        worker.run_request(
            {
                "audioPath": str(audio),
                "outputPath": str(tmp_path / "never.json"),
                "asrModelPath": str(asr_model),
                "diarizationModelPath": str(missing_diarizer),
            },
            lambda _event: None,
        )

    assert captured.value.code == "diarization_setup_required"
    assert "gated local Community-1" in str(captured.value)
    assert captured.value.details["modelId"] == worker.DIARIZATION_MODEL_ID


def test_processing_time_has_no_twenty_minute_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Advance the synthetic clock by five minutes at every measurement. The
    # mock job therefore appears to run much longer than twenty minutes while
    # still producing and preserving the complete result.
    tick = iter(value * 300.0 for value in range(10_000))
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(tick))
    audio = tmp_path / "unlimited.wav"
    _write_wav(audio, 1.0)
    events: list[dict] = []
    result = worker.run_request(
        {
            "audioPath": str(audio),
            "outputPath": str(tmp_path / "unlimited-result.json"),
            "mockInference": True,
        },
        events.append,
    )

    assert result["coverage"]["complete"] is True
    assert result["segments"]
    assert result["timings"]["totalSeconds"] > 20 * 60
    assert "deadline" not in result
    assert not any(
        isinstance(warning, dict) and warning.get("code") == "hard_deadline_exceeded"
        for warning in result["warnings"]
    )
    assert "deadlineMet" not in events[-1]


def test_mock_cli_emits_compatible_ndjson_and_result(tmp_path: Path) -> None:
    audio = tmp_path / "cli.wav"
    output = tmp_path / "cli-result.json"
    request_path = tmp_path / "request.json"
    _write_wav(audio, 1.0)
    request_path.write_text(
        json.dumps(
            {
                "protocolVersion": 1,
                "requestId": "cli-job",
                "audioPath": str(audio),
                "outputPath": str(output),
                "mockInference": True,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(WORKER_PATH), "--request-file", str(request_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert events[0]["protocolVersion"] == 1
    assert events[0]["type"] == "started"
    assert events[-1]["type"] == "completed"
    assert events[-1]["resultPath"] == str(output.resolve())
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["coverage"]["complete"] is True
    assert payload["chunks"][0]["strategy"] == "full_file"


def test_import_does_not_load_inference_frameworks() -> None:
    script = (
        "import importlib.util,sys;"
        f"p={str(WORKER_PATH)!r};"
        "s=importlib.util.spec_from_file_location('local_speech_lazy_check',p);"
        "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
        "print(','.join(str(name in sys.modules) for name in "
        "('torch','faster_whisper','ctranslate2','pyannote.audio')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False,False,False,False"
