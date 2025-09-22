"""Typer CLI entry point for ADsum."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import typer

from .config import get_settings
from .core.audio.base import AudioCapture, CaptureError, CaptureInfo
from .core.pipeline.orchestrator import RecordingOrchestrator, RecordingRequest
from .logging import configure_logging, get_logger
from .services.notes.dummy import DummyNotesService
from .services.notes.openai_notes import OpenAINotesService
from .services.transcription.dummy import DummyTranscriptionService
from .services.transcription.openai_client import OpenAITranscriptionService
from .core.audio.devices import format_device_table

app = typer.Typer(help="ADsum meeting recorder")
LOGGER = get_logger(__name__)


def _parse_device(device: Optional[str]) -> Optional[int | str]:
    if device is None:
        return None
    if device.isdigit():
        return int(device)
    return device


def _build_capture(channel: str, device: Optional[str], sample_rate: int, channels: int) -> Optional[AudioCapture]:
    if device is None:
        return None
    try:
        from .core.audio.sounddevice_backend import SoundDeviceCapture
    except ImportError as exc:  # pragma: no cover - handled by dependency guards
        raise typer.BadParameter("sounddevice dependency is required for audio capture") from exc

    capture_info = CaptureInfo(name=channel, sample_rate=sample_rate, channels=channels, device=str(device))
    try:
        return SoundDeviceCapture(info=capture_info, device=_parse_device(device))
    except CaptureError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _get_transcription_backend(name: str):
    name = name.lower()
    if name in {"", "none", "off"}:
        return None
    if name == "dummy":
        return DummyTranscriptionService()
    if name == "openai":
        return OpenAITranscriptionService()
    raise typer.BadParameter(f"Unknown transcription backend: {name}")


def _get_notes_backend(name: str):
    name = name.lower()
    if name in {"", "none", "off"}:
        return None
    if name == "dummy":
        return DummyNotesService()
    if name == "openai":
        return OpenAINotesService()
    raise typer.BadParameter(f"Unknown notes backend: {name}")


@app.command()
def devices() -> None:
    """List available audio input devices."""

    configure_logging()
    typer.echo(format_device_table())


@app.command()
def record(
    name: str = typer.Argument(..., help="Session name"),
    mic_device: Optional[str] = typer.Option(None, help="Input device id/name for microphone"),
    system_device: Optional[str] = typer.Option(None, help="Input device id/name for system audio"),
    duration: Optional[float] = typer.Option(None, help="Duration in seconds; default waits for Ctrl+C"),
    mix_down: bool = typer.Option(True, "--mix-down/--no-mix-down", help="Create a mixed track"),
    transcription_backend: str = typer.Option("dummy", help="Transcription backend: none/dummy/openai"),
    notes_backend: str = typer.Option("dummy", help="Notes backend: none/dummy/openai"),
    sample_rate: Optional[int] = typer.Option(None, help="Override sample rate"),
    channels: Optional[int] = typer.Option(None, help="Override number of channels per capture"),
) -> None:
    """Start a recording session."""

    configure_logging()
    settings = get_settings()
    sr = sample_rate or settings.sample_rate
    ch = channels or settings.channels

    captures: Dict[str, AudioCapture] = {}
    mic_capture = _build_capture("microphone", mic_device, sr, ch)
    if mic_capture is not None:
        captures["microphone"] = mic_capture
    system_capture = _build_capture("system", system_device, sr, ch)
    if system_capture is not None:
        captures["system"] = system_capture

    if not captures:
        raise typer.BadParameter("At least one audio device must be provided")

    orchestrator = RecordingOrchestrator()
    transcription = _get_transcription_backend(transcription_backend)
    notes = _get_notes_backend(notes_backend)

    request = RecordingRequest(name=name, captures=captures, mix_down=mix_down)
    outcome = orchestrator.record(request, duration=duration, transcription=transcription, notes=notes)

    typer.echo(f"Session saved at {outcome.session.id}")
    if outcome.transcripts:
        typer.echo("Transcripts:")
        for channel, transcript in outcome.transcripts.items():
            typer.echo(f"  {channel}: {len(transcript.text.split())} words")
    if outcome.notes:
        typer.echo("Notes summary:")
        typer.echo(outcome.notes.summary)


if __name__ == "__main__":  # pragma: no cover
    app()

