"""Typer CLI entry point for ADsum."""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import typer

from .config import get_settings
from .core.audio.devices import format_device_table
from .logging import configure_logging
from .ui import RecordingConsoleUI

app = typer.Typer(help="ADsum meeting recorder")


def _launch_ui(
    *,
    name: Optional[str],
    mic_device: Optional[str],
    system_device: Optional[str],
    mix_down: bool,
    transcription_backend: str,
    notes_backend: str,
    sample_rate: Optional[int],
    channels: Optional[int],
    interface: str,
) -> None:
    configure_logging()
    settings = get_settings()
    ui_kwargs = dict(
        settings=settings,
        sample_rate=sample_rate,
        channels=channels,
        mix_down=mix_down,
        default_name=name,
        default_mic=mic_device,
        default_system=system_device,
        transcription_backend_name=transcription_backend,
        notes_backend_name=notes_backend,
    )

    ui_instance = _resolve_ui(interface, ui_kwargs)
    ui_instance.run()


def _resolve_ui(interface: str, ui_kwargs: Dict[str, Any]):
    """Return the UI implementation based on the requested interface."""

    normalized = (interface or "auto").lower()

    if normalized not in {"auto", "console", "window"}:
        raise typer.BadParameter("Interface must be one of: auto, console, window")

    if normalized == "auto":
        normalized = "window" if sys.platform.startswith("win") else "console"

    if normalized == "window":
        try:
            from .ui import RecordingWindowUI

            if RecordingWindowUI.is_supported():
                return RecordingWindowUI(**ui_kwargs)
            typer.echo(
                "Window UI is not available on this system. Falling back to console interface.",
                err=True,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback
            typer.echo(
                f"Failed to initialise the window UI ({exc}). Falling back to console.",
                err=True,
            )

    return RecordingConsoleUI(**ui_kwargs)


@app.command()
def devices() -> None:
    """List available audio input devices."""

    configure_logging()
    typer.echo(format_device_table())


@app.command()
def ui(
    name: Optional[str] = typer.Argument(None, help="Optional session name to pre-fill"),
    mic_device: Optional[str] = typer.Option(None, help="Default microphone device id/name"),
    system_device: Optional[str] = typer.Option(None, help="Default system audio device id/name"),
    mix_down: bool = typer.Option(True, "--mix-down/--no-mix-down", help="Create a mixed track"),
    transcription_backend: str = typer.Option(
        "dummy", help="Transcription backend to pre-select: none/dummy/openai"
    ),
    notes_backend: str = typer.Option(
        "dummy", help="Notes backend to pre-select: none/dummy/openai"
    ),
    sample_rate: Optional[int] = typer.Option(None, help="Override sample rate"),
    channels: Optional[int] = typer.Option(None, help="Override number of channels"),
    interface: str = typer.Option(
        "auto",
        "--interface",
        "-i",
        help="User interface to launch: auto (default), console, or window.",
    ),
) -> None:
    """Launch the interactive UI without starting a recording directly."""

    _launch_ui(
        name=name,
        mic_device=mic_device,
        system_device=system_device,
        mix_down=mix_down,
        transcription_backend=transcription_backend,
        notes_backend=notes_backend,
        sample_rate=sample_rate,
        channels=channels,
        interface=interface,
    )


@app.command()
def record(
    name: Optional[str] = typer.Argument(None, help="Optional session name to pre-fill"),
    mic_device: Optional[str] = typer.Option(None, help="Default microphone device id/name"),
    system_device: Optional[str] = typer.Option(None, help="Default system audio device id/name"),
    mix_down: bool = typer.Option(True, "--mix-down/--no-mix-down", help="Create a mixed track"),
    transcription_backend: str = typer.Option(
        "dummy", help="Transcription backend to pre-select: none/dummy/openai"
    ),
    notes_backend: str = typer.Option(
        "dummy", help="Notes backend to pre-select: none/dummy/openai"
    ),
    sample_rate: Optional[int] = typer.Option(None, help="Override sample rate"),
    channels: Optional[int] = typer.Option(None, help="Override number of channels"),
    interface: str = typer.Option(
        "auto",
        "--interface",
        "-i",
        help="User interface to launch: auto (default), console, or window.",
    ),
) -> None:
    """Backward-compatible alias that now launches the interactive UI."""

    typer.echo("Launching ADsum UI. Recording control is now handled interactively.")
    _launch_ui(
        name=name,
        mic_device=mic_device,
        system_device=system_device,
        mix_down=mix_down,
        transcription_backend=transcription_backend,
        notes_backend=notes_backend,
        sample_rate=sample_rate,
        channels=channels,
        interface=interface,
    )


if __name__ == "__main__":  # pragma: no cover
    app()

