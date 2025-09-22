"""Data models used by ADsum."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptResult(BaseModel):
    session_id: str
    channel: str
    text: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    raw_response: Optional[dict] = None


class NoteDocument(BaseModel):
    session_id: str
    title: str
    summary: str
    action_items: List[str] = Field(default_factory=list)


@dataclass
class RecordingSession:
    id: str
    name: str
    created_at: float
    duration: float
    sample_rate: int
    channels: int
    audio_paths: Dict[str, Path]
    mix_path: Optional[Path] = None


__all__ = [
    "TranscriptSegment",
    "TranscriptResult",
    "NoteDocument",
    "RecordingSession",
]

