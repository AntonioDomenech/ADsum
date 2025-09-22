"""Dummy notes generator for offline usage."""

from __future__ import annotations

from typing import Iterable

from ...data.models import NoteDocument, RecordingSession, TranscriptResult
from .base import NotesService


class DummyNotesService(NotesService):
    def generate_notes(
        self, session: RecordingSession, transcripts: Iterable[TranscriptResult]
    ) -> NoteDocument:
        combined = "\n".join(transcript.text for transcript in transcripts)
        summary = combined[:280] + ("..." if len(combined) > 280 else "")
        return NoteDocument(
            session_id=session.id,
            title=f"Notes for {session.name}",
            summary=summary or "No transcript available.",
            action_items=[],
        )


__all__ = ["DummyNotesService"]

