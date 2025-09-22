"""Recording orchestrator coordinating capture, storage, and processing."""

from __future__ import annotations

import contextlib
import time
from threading import Event
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from ...config import get_settings
from ...data.models import NoteDocument, RecordingSession, TranscriptResult
from ...data.storage import SessionStore
from ...logging import get_logger
from ...services.notes.base import NotesService
from ...services.transcription.base import TranscriptionService
from ...utils.audio import mix_audio_files
from ..audio.base import AudioCapture
from ..audio.writers import AudioFileWriter

LOGGER = get_logger(__name__)


@dataclass
class RecordingRequest:
    name: str
    captures: Dict[str, AudioCapture]
    mix_down: bool = True
    session_id: Optional[str] = None


@dataclass
class RecordingOutcome:
    session: RecordingSession
    transcripts: Dict[str, TranscriptResult] = field(default_factory=dict)
    notes: Optional[NoteDocument] = None


class RecordingControl:
    """Runtime control flags for long-running recording sessions."""

    def __init__(self) -> None:
        self._stop = Event()
        self._resume = Event()
        self._resume.set()

    def request_stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def request_pause(self) -> None:
        if not self._stop.is_set():
            self._resume.clear()

    def request_resume(self) -> None:
        if not self._stop.is_set():
            self._resume.set()

    @property
    def should_stop(self) -> bool:
        return self._stop.is_set()

    @property
    def is_paused(self) -> bool:
        return not self._resume.is_set() and not self._stop.is_set()

    @property
    def is_recording(self) -> bool:
        return self._resume.is_set() and not self._stop.is_set()

    def wait_for_resume(self, timeout: float = 0.1) -> None:
        self._resume.wait(timeout)


class RecordingOrchestrator:
    """High-level coordinator for recording sessions."""

    def __init__(self, base_dir: Optional[Path] = None, store: Optional[SessionStore] = None) -> None:
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.base_dir)
        self.store = store or SessionStore(settings.database_path)
        self.store.initialize()

    def _create_session_dirs(self, session_id: str) -> Dict[str, Path]:
        base = self.base_dir / session_id
        raw = base / "raw"
        processed = base / "processed"
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        return {"base": base, "raw": raw, "processed": processed}

    def record(
        self,
        request: RecordingRequest,
        duration: Optional[float] = None,
        transcription: Optional[TranscriptionService] = None,
        notes: Optional[NotesService] = None,
        control: Optional[RecordingControl] = None,
        transcript_callback: Optional[Callable[[TranscriptResult], None]] = None,
        transcript_update_callback: Optional[Callable[[TranscriptResult], None]] = None,
    ) -> RecordingOutcome:
        if not request.captures:
            raise ValueError("At least one capture channel must be configured")

        session_id = request.session_id or f"{get_settings().session_prefix}-{uuid.uuid4().hex[:8]}"
        dirs = self._create_session_dirs(session_id)
        writers: Dict[str, AudioFileWriter] = {}
        capture_paths: Dict[str, Path] = {}

        LOGGER.info("Starting recording session %s", session_id)
        start_time = time.monotonic()

        try:
            for channel, capture in request.captures.items():
                file_path = dirs["raw"] / f"{channel}.wav"
                capture.start()
                info = capture.info
                writers[channel] = AudioFileWriter(file_path, info.sample_rate, info.channels)
                capture_paths[channel] = file_path

            while True:
                if control and control.should_stop:
                    break

                if control and control.is_paused:
                    control.wait_for_resume(timeout=0.1)
                    continue

                active = False
                for channel, capture in request.captures.items():
                    chunk = capture.read(timeout=0.1)
                    if chunk is not None:
                        writers[channel].write(np.asarray(chunk, dtype=np.float32))
                        active = True
                if duration is not None and time.monotonic() - start_time >= duration:
                    break
                if control and control.should_stop:
                    break
                if not active and duration is None:
                    if control:
                        control.wait_for_resume(timeout=0.1)
                    continue
            LOGGER.info("Recording duration reached; stopping streams")
        except KeyboardInterrupt:
            LOGGER.info("Recording interrupted by user; finishing up")
        finally:
            for channel, capture in request.captures.items():
                with contextlib.suppress(Exception):
                    capture.stop()
                # Drain remaining chunks without blocking to ensure audio is flushed
                writer = writers.get(channel)
                if writer is None:
                    continue
                while True:
                    chunk = capture.read(timeout=0)
                    if chunk is None:
                        break
                    writer.write(np.asarray(chunk, dtype=np.float32))
                with contextlib.suppress(Exception):
                    capture.close()
            for writer in writers.values():
                writer.close()

        duration_seconds = max((writer.duration_seconds for writer in writers.values()), default=0.0)
        session = RecordingSession(
            id=session_id,
            name=request.name,
            created_at=time.time(),
            duration=duration_seconds,
            sample_rate=max((capture.info.sample_rate for capture in request.captures.values()), default=0),
            channels=max((capture.info.channels for capture in request.captures.values()), default=0),
            audio_paths=capture_paths,
            mix_path=None,
        )
        self.store.save_session(session)

        mix_path: Optional[Path] = None
        if request.mix_down and len(capture_paths) >= 1:
            mix_path = dirs["processed"] / "mix.wav"
            try:
                mix_audio_files(list(capture_paths.values()), mix_path)
                session.mix_path = mix_path
                self.store.update_mix_path(session.id, mix_path)
            except Exception as exc:  # pragma: no cover - exercised only with problematic files
                LOGGER.exception("Failed to mix down audio: %s", exc)

        transcripts: Dict[str, TranscriptResult] = {}
        if transcription is not None and (mix_path or capture_paths):
            targets = [mix_path] if mix_path else list(capture_paths.values())
            for path in targets:
                if path is None:
                    continue
                LOGGER.info("Transcribing %s", path)
                latest_update: Optional[TranscriptResult] = None

                def _handle_update(update: TranscriptResult) -> None:
                    nonlocal latest_update
                    latest_update = update
                    if transcript_update_callback is not None:
                        try:
                            transcript_update_callback(update)
                        except Exception:  # pragma: no cover - callbacks should not break pipeline
                            LOGGER.exception("Transcript update callback raised an exception")

                result = transcription.transcribe_stream(
                    session,
                    path,
                    on_update=_handle_update if transcript_update_callback is not None else None,
                )
                if transcript_update_callback is not None:
                    if latest_update is None or latest_update.model_dump() != result.model_dump():
                        _handle_update(result)
                transcripts[result.channel] = result
                self.store.save_transcript(result)
                if transcript_callback is not None:
                    try:
                        transcript_callback(result)
                    except Exception:  # pragma: no cover - callbacks should not break pipeline
                        LOGGER.exception("Transcript callback raised an exception")

        note_document: Optional[NoteDocument] = None
        if notes is not None and transcripts:
            LOGGER.info("Generating notes for session %s", session.id)
            note_document = notes.generate_notes(session, list(transcripts.values()))
            self.store.save_notes(note_document)

        return RecordingOutcome(session=session, transcripts=transcripts, notes=note_document)


__all__ = [
    "RecordingControl",
    "RecordingOrchestrator",
    "RecordingRequest",
    "RecordingOutcome",
]

