"""Notes generation service abstractions."""

from __future__ import annotations

import abc
from typing import Iterable

from ...data.models import NoteDocument, RecordingSession, TranscriptResult


class NotesService(abc.ABC):
    @abc.abstractmethod
    def generate_notes(
        self, session: RecordingSession, transcripts: Iterable[TranscriptResult]
    ) -> NoteDocument:
        raise NotImplementedError


__all__ = ["NotesService"]

