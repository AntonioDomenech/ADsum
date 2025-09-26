from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from adsum.data.models import RecordingSession, TranscriptResult
from adsum.services.transcription.openai_client import OpenAITranscriptionService
from adsum.utils.audio import AudioChunk


class DummyResponseFormatError(Exception):
    """Fake error raised by the mocked OpenAI client for unsupported formats."""


def _make_service() -> tuple[OpenAITranscriptionService, list[str]]:
    service = object.__new__(OpenAITranscriptionService)
    service.model = "test-model"
    service._openai_error_cls = DummyResponseFormatError
    service.max_upload_bytes = 1024 * 1024 * 1024

    calls: list[str] = []

    class DummyTranscriptions:
        def create(self, model, file, response_format):
            calls.append(response_format)
            if response_format != "text":
                raise DummyResponseFormatError(
                    f"response_format '{response_format}' unsupported"
                )
            return "Mock transcript from text response"

    transcriptions = DummyTranscriptions()
    service.client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=transcriptions)
    )
    return service, calls


def test_transcribe_falls_back_to_text(tmp_path):
    service, calls = _make_service()

    audio_path = tmp_path / "example.wav"
    audio_path.write_bytes(b"fake audio content")

    session = RecordingSession(
        id="session-id",
        name="Example",
        created_at=0,
        duration=0,
        sample_rate=16000,
        channels=1,
        audio_paths={"example": audio_path},
    )

    result = service.transcribe(session, audio_path)

    assert calls == ["verbose_json", "json", "text"]
    assert result.text == "Mock transcript from text response"
    assert result.segments == []
    assert result.raw_response == {"text": "Mock transcript from text response"}


def test_transcribe_large_file_chunking(monkeypatch, tmp_path):
    service, _ = _make_service()

    calls: list[tuple[str, str, str]] = []

    class DummyTranscriptions:
        def create(self, model, file, response_format):
            calls.append((model, Path(file.name).name, response_format))
            return {
                "text": f"chunk-{len(calls)}",
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": f"segment-{len(calls)}"},
                ],
            }

    service.client.audio.transcriptions = DummyTranscriptions()
    service.max_upload_bytes = 10

    chunk1 = tmp_path / "chunk1.wav"
    chunk2 = tmp_path / "chunk2.wav"
    chunk1.write_bytes(b"audio-1")
    chunk2.write_bytes(b"audio-2")

    chunks = [
        AudioChunk(path=chunk1, start=0.0, duration=1.0),
        AudioChunk(path=chunk2, start=1.0, duration=1.0),
    ]

    def fake_split(path, max_bytes):  # noqa: ARG001
        return chunks.copy()

    monkeypatch.setattr("adsum.services.transcription.openai_client.split_wave_file", fake_split)

    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"large-audio")

    session = RecordingSession(
        id="session-id",
        name="Example",
        created_at=0,
        duration=0,
        sample_rate=16000,
        channels=1,
        audio_paths={"example": audio_path},
    )

    result = service.transcribe(session, audio_path)

    assert [call[1] for call in calls] == ["chunk1.wav", "chunk2.wav"]
    assert result.text == "chunk-1\nchunk-2"
    assert [segment.text for segment in result.segments] == ["segment-1", "segment-2"]
    assert result.segments[0].start == 0.0
    assert result.segments[1].start == 1.0
    assert not chunk1.exists()
    assert not chunk2.exists()


def test_transcribe_stream_chunking(monkeypatch, tmp_path):
    service, _ = _make_service()
    updates: list[TranscriptResult] = []

    class DummyTranscriptions:
        def create(self, model, file, response_format):
            return {
                "text": f"chunk-{len(updates) + 1}",
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": f"segment-{len(updates) + 1}"},
                ],
            }

    service.client.audio.transcriptions = DummyTranscriptions()
    service.max_upload_bytes = 10

    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"audio")

    chunks = [
        AudioChunk(path=chunk, start=0.0, duration=1.0),
    ]

    def fake_split(path, max_bytes):  # noqa: ARG001
        return chunks.copy()

    monkeypatch.setattr("adsum.services.transcription.openai_client.split_wave_file", fake_split)

    audio_path = tmp_path / "recording.wav"
    audio_path.write_bytes(b"large-audio")

    session = RecordingSession(
        id="session-id",
        name="Example",
        created_at=0,
        duration=0,
        sample_rate=16000,
        channels=1,
        audio_paths={"example": audio_path},
    )

    def on_update(result: TranscriptResult) -> None:
        updates.append(result)

    final = service.transcribe_stream(session, audio_path, on_update=on_update)

    assert updates
    assert final.text == "chunk-1"
    assert final.segments[0].start == 0.0
    assert not chunk.exists()
