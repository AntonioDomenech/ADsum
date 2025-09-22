"""Transcription service abstractions."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Callable, Optional

from ...data.models import RecordingSession, TranscriptResult


class TranscriptionService(abc.ABC):
    """Convert audio into transcript results."""

    @abc.abstractmethod
    def transcribe(self, session: RecordingSession, audio_path: Path) -> TranscriptResult:
        raise NotImplementedError

    def transcribe_stream(
        self,
        session: RecordingSession,
        audio_path: Path,
        on_update: Optional[Callable[[TranscriptResult], None]] = None,
    ) -> TranscriptResult:
        """Transcribe audio while optionally emitting incremental updates."""

        return self.transcribe(session, audio_path)


__all__ = ["TranscriptionService"]

