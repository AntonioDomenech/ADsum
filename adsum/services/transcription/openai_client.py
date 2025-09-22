"""OpenAI powered transcription service."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ...config import get_settings
from ...data.models import RecordingSession, TranscriptResult, TranscriptSegment
from ...logging import get_logger
from .base import TranscriptionService

LOGGER = get_logger(__name__)


class OpenAITranscriptionService(TranscriptionService):
    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.model = model or settings.openai_transcription_model
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("openai package is required for OpenAITranscriptionService") from exc
        self.client = OpenAI()

    def transcribe(self, session: RecordingSession, audio_path: Path) -> TranscriptResult:
        LOGGER.info("Requesting OpenAI transcription for %s", audio_path)
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="verbose_json",
            )
        segments = [
            TranscriptSegment(start=segment["start"], end=segment["end"], text=segment["text"])
            for segment in response.get("segments", [])
        ]
        text = response.get("text", "")
        return TranscriptResult(
            session_id=session.id,
            channel=audio_path.stem,
            text=text,
            segments=segments,
            raw_response=response,
        )


__all__ = ["OpenAITranscriptionService"]

