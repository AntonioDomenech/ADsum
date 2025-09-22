"""Dummy transcription service for testing or offline usage."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from ...data.models import RecordingSession, TranscriptResult, TranscriptSegment
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
            segments=[TranscriptSegment(text=text)],
        )

    def transcribe_stream(
        self,
        session: RecordingSession,
        audio_path: Path,
        on_update: Optional[Callable[[TranscriptResult], None]] = None,
    ) -> TranscriptResult:
        result = self.transcribe(session, audio_path)
        if on_update is not None:
            preview_text = result.text.split(". ")[0].strip()
            if preview_text:
                preview = TranscriptResult(
                    session_id=session.id,
                    channel=self.channel_name,
                    text=preview_text,
                    segments=[TranscriptSegment(text=preview_text)],
                )
                try:
                    on_update(preview)
                except Exception:  # pragma: no cover - defensive guard for callbacks
                    pass
        return result


__all__ = ["DummyTranscriptionService"]

