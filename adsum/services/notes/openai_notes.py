"""OpenAI-powered note summarisation."""

from __future__ import annotations

from typing import Iterable, Optional

from ...config import get_settings
from ...data.models import NoteDocument, RecordingSession, TranscriptResult
from ...logging import get_logger
from .base import NotesService

LOGGER = get_logger(__name__)


class OpenAINotesService(NotesService):
    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.model = model or settings.openai_notes_model
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("openai package is required for OpenAINotesService") from exc
        self.client = OpenAI()

    def generate_notes(
        self, session: RecordingSession, transcripts: Iterable[TranscriptResult]
    ) -> NoteDocument:
        transcript_text = "\n".join(
            f"[{t.channel}] {t.text}" for t in transcripts if t.text
        )
        LOGGER.info("Requesting OpenAI notes for session %s", session.id)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": "Summarise the meeting transcript, list key points and action items",
                },
                {
                    "role": "user",
                    "content": transcript_text,
                },
            ],
        )
        summary = response.output_text
        action_items = []
        return NoteDocument(
            session_id=session.id,
            title=f"Notes for {session.name}",
            summary=summary,
            action_items=action_items,
        )


__all__ = ["OpenAINotesService"]

