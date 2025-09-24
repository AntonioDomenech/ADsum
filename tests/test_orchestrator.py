import queue
import threading
import time
from pathlib import Path

import numpy as np

import adsum.core.pipeline.orchestrator as orchestrator_module
from adsum.core.audio.base import AudioCapture, CaptureInfo
from adsum.core.pipeline.orchestrator import (
    RecordingControl,
    RecordingOrchestrator,
    RecordingRequest,
)
from adsum.data.models import RecordingSession, TranscriptResult
from adsum.data.storage import SessionStore
from adsum.services.notes.dummy import DummyNotesService
from adsum.services.transcription.base import TranscriptionService
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


class StreamingCapture(AudioCapture):
    def __init__(self, info: CaptureInfo, chunk: np.ndarray) -> None:
        self.info = info
        self._chunk = chunk
        self._stopped = False

    def start(self) -> None:
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def close(self) -> None:
        self._stopped = True

    def read(self, timeout: float | None = None):
        if self._stopped:
            return None
        if timeout and timeout > 0:
            time.sleep(min(timeout, 0.01))
        return self._chunk


class PathTrackingTranscriptionService(TranscriptionService):
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def transcribe(
        self, session: RecordingSession, audio_path: Path
    ) -> TranscriptResult:
        self.paths.append(audio_path)
        text = f"Transcript for {audio_path.stem}"
        return TranscriptResult(
            session_id=session.id,
            channel=audio_path.stem,
            text=text,
        )


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
    sessions = orchestrator.store.list_sessions()
    assert any(session.id == outcome.session.id for session in sessions)


def test_orchestrator_uses_raw_targets_when_mix_fails(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "adsum.db"
    store = SessionStore(db_path)
    orchestrator = RecordingOrchestrator(base_dir=tmp_path / "recordings", store=store)

    sample_rate = 16000
    chunk = np.zeros((sample_rate // 10, 1), dtype=np.float32)
    captures = {
        "microphone": FakeCapture(
            CaptureInfo(name="microphone", sample_rate=sample_rate, channels=1),
            [chunk],
        ),
        "system": FakeCapture(
            CaptureInfo(name="system", sample_rate=sample_rate, channels=1),
            [chunk],
        ),
    }

    request = RecordingRequest(name="Mix Failure", captures=captures)

    def _raise_mix_error(*_args, **_kwargs):
        raise RuntimeError("mix failed")

    monkeypatch.setattr(
        "adsum.core.pipeline.orchestrator.mix_audio_files", _raise_mix_error
    )

    transcription = PathTrackingTranscriptionService()

    outcome = orchestrator.record(
        request,
        duration=0.05,
        transcription=transcription,
    )

    assert outcome.session.mix_path is None
    recorded_stems = {path.stem for path in transcription.paths}
    assert recorded_stems == {"microphone", "system"}
    assert set(outcome.transcripts.keys()) == recorded_stems


def test_orchestrator_reports_silent_channels(tmp_path: Path) -> None:
    db_path = tmp_path / "adsum.db"
    store = SessionStore(db_path)
    orchestrator = RecordingOrchestrator(base_dir=tmp_path / "recordings", store=store)

    sample_rate = 16000
    chunk = np.zeros((sample_rate // 10, 1), dtype=np.float32)
    captures = {
        "microphone": FakeCapture(
            CaptureInfo(name="microphone", sample_rate=sample_rate, channels=1),
            [chunk],
        ),
        "system": FakeCapture(
            CaptureInfo(name="system", sample_rate=sample_rate, channels=1),
            [],
        ),
    }

    request = RecordingRequest(name="Silent Channel", captures=captures, mix_down=False)
    transcription = PathTrackingTranscriptionService()

    outcome = orchestrator.record(
        request,
        duration=0.05,
        transcription=transcription,
    )

    metrics = outcome.channel_metrics
    assert metrics["microphone"].frames > 0
    assert metrics["system"].frames == 0
    assert metrics["system"].is_silent
    assert transcription.paths == [outcome.session.audio_paths["microphone"]]


def test_orchestrator_skips_silent_sources_in_mix(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "adsum.db"
    store = SessionStore(db_path)
    orchestrator = RecordingOrchestrator(base_dir=tmp_path / "recordings", store=store)

    sample_rate = 16000
    chunk = np.zeros((sample_rate // 10, 1), dtype=np.float32)
    captures = {
        "microphone": FakeCapture(
            CaptureInfo(name="microphone", sample_rate=sample_rate, channels=1),
            [chunk],
        ),
        "system": FakeCapture(
            CaptureInfo(name="system", sample_rate=sample_rate, channels=1),
            [],
        ),
    }

    recorded_sources: list[list[Path]] = []
    original_mix = orchestrator_module.mix_audio_files

    def _tracking_mix(paths, output_path):
        materialised = list(paths)
        recorded_sources.append(materialised)
        return original_mix(materialised, output_path)

    monkeypatch.setattr(orchestrator_module, "mix_audio_files", _tracking_mix)

    outcome = orchestrator.record(
        RecordingRequest(name="Mix Silent", captures=captures),
        duration=0.05,
        transcription=DummyTranscriptionService(),
    )

    assert recorded_sources, "mix_audio_files should be invoked"
    assert len(recorded_sources[0]) == 1
    assert recorded_sources[0][0].name == "microphone.wav"
    assert outcome.session.mix_path is not None
    assert outcome.channel_metrics["system"].is_silent


def test_orchestrator_emits_transcript_updates(tmp_path: Path) -> None:
    db_path = tmp_path / "adsum.db"
    store = SessionStore(db_path)
    orchestrator = RecordingOrchestrator(base_dir=tmp_path / "recordings", store=store)

    sample_rate = 16000
    chunk = np.zeros((sample_rate // 10, 1), dtype=np.float32)
    capture_info = CaptureInfo(name="microphone", sample_rate=sample_rate, channels=1)
    capture = FakeCapture(capture_info, [chunk])

    received = []
    request = RecordingRequest(name="Callback Session", captures={"microphone": capture})
    orchestrator.record(
        request,
        duration=0.02,
        transcription=DummyTranscriptionService(channel_name="microphone"),
        transcript_callback=received.append,
    )

    assert received, "transcript callback should be invoked"
    assert received[0].channel == "microphone"
    assert "Dummy transcript" in received[0].text


def test_orchestrator_streaming_callback(tmp_path: Path) -> None:
    db_path = tmp_path / "adsum.db"
    store = SessionStore(db_path)
    orchestrator = RecordingOrchestrator(base_dir=tmp_path / "recordings", store=store)

    sample_rate = 16000
    chunk = np.zeros((sample_rate // 10, 1), dtype=np.float32)
    capture_info = CaptureInfo(name="microphone", sample_rate=sample_rate, channels=1)
    capture = FakeCapture(capture_info, [chunk])

    updates: list[TranscriptResult] = []
    finals: list[TranscriptResult] = []
    request = RecordingRequest(name="Streaming Session", captures={"microphone": capture})
    orchestrator.record(
        request,
        duration=0.02,
        transcription=DummyTranscriptionService(channel_name="microphone"),
        transcript_update_callback=updates.append,
        transcript_callback=finals.append,
    )

    assert updates, "streaming callback should receive at least one update"
    assert finals, "final callback should still be invoked"
    assert updates[-1].text == finals[0].text
    assert any(update.text != finals[0].text for update in updates)


def test_orchestrator_respects_recording_control(tmp_path: Path) -> None:
    db_path = tmp_path / "adsum.db"
    store = SessionStore(db_path)
    orchestrator = RecordingOrchestrator(base_dir=tmp_path / "recordings", store=store)

    sample_rate = 8000
    chunk = np.zeros((sample_rate // 20, 1), dtype=np.float32)
    capture_info = CaptureInfo(name="loop", sample_rate=sample_rate, channels=1)
    capture = StreamingCapture(capture_info, chunk)

    request = RecordingRequest(name="Controlled Session", captures={"loop": capture})
    control = RecordingControl()

    stopper = threading.Timer(0.05, control.request_stop)
    stopper.start()
    try:
        outcome = orchestrator.record(request, control=control)
    finally:
        stopper.cancel()

    assert outcome.session.duration > 0
    assert orchestrator.store.fetch_session(outcome.session.id) is not None
