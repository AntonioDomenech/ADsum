from __future__ import annotations

from types import SimpleNamespace

from adsum.data.models import RecordingSession
from adsum.services.transcription.openai_client import OpenAITranscriptionService


class DummyResponseFormatError(Exception):
    """Fake error raised by the mocked OpenAI client for unsupported formats."""


def _make_service() -> tuple[OpenAITranscriptionService, list[str]]:
    service = object.__new__(OpenAITranscriptionService)
    service.model = "test-model"
    service._openai_error_cls = DummyResponseFormatError

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
