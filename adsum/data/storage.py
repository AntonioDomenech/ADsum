"""SQLite storage helpers for sessions, transcripts, and notes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import NoteDocument, RecordingSession, TranscriptResult, TranscriptSegment


class SessionStore:
    """Persistent storage built on SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    duration REAL NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    audio_paths TEXT NOT NULL,
                    mix_path TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    text TEXT NOT NULL,
                    segments TEXT,
                    raw_response TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    action_items TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            conn.commit()

    def save_session(self, session: RecordingSession) -> None:
        audio_paths = {k: str(v) for k, v in session.audio_paths.items()}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    id, name, created_at, duration, sample_rate, channels, audio_paths, mix_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.name,
                    session.created_at,
                    session.duration,
                    session.sample_rate,
                    session.channels,
                    json.dumps(audio_paths),
                    str(session.mix_path) if session.mix_path else None,
                ),
            )
            conn.commit()

    def update_mix_path(self, session_id: str, mix_path: Path) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET mix_path = ? WHERE id = ?",
                (str(mix_path), session_id),
            )
            conn.commit()

    def save_transcript(self, result: TranscriptResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transcripts (session_id, channel, text, segments, raw_response)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.session_id,
                    result.channel,
                    result.text,
                    json.dumps([segment.model_dump() for segment in result.segments]),
                    json.dumps(result.raw_response) if result.raw_response else None,
                ),
            )
            conn.commit()

    def save_notes(self, notes: NoteDocument) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO notes (session_id, title, summary, action_items)
                VALUES (?, ?, ?, ?)
                """,
                (
                    notes.session_id,
                    notes.title,
                    notes.summary,
                    json.dumps(notes.action_items),
                ),
            )
            conn.commit()

    def fetch_session(self, session_id: str) -> Optional[RecordingSession]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, created_at, duration, sample_rate, channels, audio_paths, mix_path FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        audio_paths = {k: Path(v) for k, v in json.loads(row[6]).items()}
        mix_path = Path(row[7]) if row[7] else None
        return RecordingSession(
            id=row[0],
            name=row[1],
            created_at=row[2],
            duration=row[3],
            sample_rate=row[4],
            channels=row[5],
            audio_paths=audio_paths,
            mix_path=mix_path,
        )

    def fetch_transcripts(self, session_id: str) -> List[TranscriptResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT channel, text, segments, raw_response FROM transcripts WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        results: List[TranscriptResult] = []
        for channel, text, segments_json, raw_json in rows:
            segments_data = json.loads(segments_json) if segments_json else []
            segments = [TranscriptSegment(**segment) for segment in segments_data]
            raw = json.loads(raw_json) if raw_json else None
            results.append(
                TranscriptResult(
                    session_id=session_id,
                    channel=channel,
                    text=text,
                    segments=segments,
                    raw_response=raw,
                )
            )
        return results

    def fetch_notes(self, session_id: str) -> Optional[NoteDocument]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT title, summary, action_items FROM notes WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        action_items = json.loads(row[2]) if row[2] else []
        return NoteDocument(
            session_id=session_id,
            title=row[0],
            summary=row[1],
            action_items=action_items,
        )


__all__ = ["SessionStore"]

