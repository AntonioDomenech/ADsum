"""Transcription service abstractions."""

from __future__ import annotations

import abc
from pathlib import Path

from ...data.models import RecordingSession, TranscriptResult


class TranscriptionService(abc.ABC):
    """Convert audio into transcript results."""

    @abc.abstractmethod
    def transcribe(self, session: RecordingSession, audio_path: Path) -> TranscriptResult:
        raise NotImplementedError


__all__ = ["TranscriptionService"]

