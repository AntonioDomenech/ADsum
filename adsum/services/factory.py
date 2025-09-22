"""Factories for runtime service selection."""

from __future__ import annotations

from typing import Optional

from .notes.base import NotesService
from .notes.dummy import DummyNotesService
from .notes.openai_notes import OpenAINotesService
from .transcription.base import TranscriptionService
from .transcription.dummy import DummyTranscriptionService
from .transcription.openai_client import OpenAITranscriptionService


class ServiceConfigurationError(ValueError):
    """Raised when an unknown backend is requested."""


def _normalise(name: Optional[str]) -> str:
    if not name:
        return "none"
    return name.strip().lower()


def resolve_transcription_backend(name: Optional[str]) -> Optional[TranscriptionService]:
    backend = _normalise(name)
    if backend in {"", "none", "off"}:
        return None
    if backend == "dummy":
        return DummyTranscriptionService()
    if backend == "openai":
        return OpenAITranscriptionService()
    raise ServiceConfigurationError(f"Unknown transcription backend: {name}")


def resolve_notes_backend(name: Optional[str]) -> Optional[NotesService]:
    backend = _normalise(name)
    if backend in {"", "none", "off"}:
        return None
    if backend == "dummy":
        return DummyNotesService()
    if backend == "openai":
        return OpenAINotesService()
    raise ServiceConfigurationError(f"Unknown notes backend: {name}")


__all__ = [
    "ServiceConfigurationError",
    "resolve_notes_backend",
    "resolve_transcription_backend",
]

