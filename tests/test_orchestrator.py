import queue
from pathlib import Path

import numpy as np

from adsum.core.audio.base import AudioCapture, CaptureInfo
from adsum.core.pipeline.orchestrator import RecordingOrchestrator, RecordingRequest
from adsum.data.storage import SessionStore
from adsum.services.notes.dummy import DummyNotesService
from adsum.services.transcription.dummy import DummyTranscriptionService


class FakeCapture(AudioCapture):
    def __init__(self, info: CaptureInfo, chunks: list[np.ndarray]) -> None:
        self.info = info
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        for chunk in chunks:
            self._queue.put(chunk)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self, timeout: float | None = None):
        try:
            if timeout is None or timeout <= 0:
                return self._queue.get_nowait()
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def test_orchestrator_record_pipeline(tmp_path: Path) -> None:
    db_path = tmp_path / "adsum.db"
    store = SessionStore(db_path)
    orchestrator = RecordingOrchestrator(base_dir=tmp_path / "recordings", store=store)

    sample_rate = 16000
    chunk = np.zeros((sample_rate // 10, 1), dtype=np.float32)
    capture_info = CaptureInfo(name="microphone", sample_rate=sample_rate, channels=1)
    capture = FakeCapture(capture_info, [chunk, chunk])

    request = RecordingRequest(name="Test Session", captures={"microphone": capture})
    outcome = orchestrator.record(
        request,
        duration=0.05,
        transcription=DummyTranscriptionService(),
        notes=DummyNotesService(),
    )

    assert outcome.session.id
    for path in outcome.session.audio_paths.values():
        assert path.exists() and path.stat().st_size > 0
    assert orchestrator.store.fetch_session(outcome.session.id) is not None
    transcripts = orchestrator.store.fetch_transcripts(outcome.session.id)
    assert transcripts and transcripts[0].text
    notes = orchestrator.store.fetch_notes(outcome.session.id)
    assert notes is not None
