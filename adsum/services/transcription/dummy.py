"""Dummy transcription service for testing or offline usage."""

from __future__ import annotations

from pathlib import Path

from ...data.models import RecordingSession, TranscriptResult
from .base import TranscriptionService


class DummyTranscriptionService(TranscriptionService):
    def __init__(self, channel_name: str = "mixed") -> None:
        self.channel_name = channel_name

    def transcribe(self, session: RecordingSession, audio_path: Path) -> TranscriptResult:
        text = (
            f"Dummy transcript for session '{session.name}' from {audio_path.name}. "
            "Replace with a real transcription backend."
        )
        return TranscriptResult(
            session_id=session.id,
            channel=self.channel_name,
            text=text,
        )


__all__ = ["DummyTranscriptionService"]

