"""Typer CLI entry point for ADsum."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, TypeVar, Union, cast

import typer

from .config import get_settings
from .core.audio.devices import format_device_table
from .logging import configure_logging
from .ui import RecordingConsoleUI

app = typer.Typer(help="ADsum meeting recorder")


class BackendChoice(str, Enum):
    NONE = "none"
    DUMMY = "dummy"
    OPENAI = "openai"


class InterfaceChoice(str, Enum):
    AUTO = "auto"
    CONSOLE = "console"
    WINDOW = "window"


@dataclass
class UILaunchConfig:
    settings: Any
    sample_rate: Optional[int]
    channels: Optional[int]
    mix_down: bool
    default_name: Optional[str]
    default_mic: Optional[str]
    default_system: Optional[str]
    transcription_backend_name: BackendChoice
    notes_backend_name: BackendChoice

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "mix_down": self.mix_down,
            "default_name": self.default_name,
            "default_mic": self.default_mic,
            "default_system": self.default_system,
            "transcription_backend_name": self.transcription_backend_name.value,
            "notes_backend_name": self.notes_backend_name.value,
        }


SESSION_NAME_ARGUMENT = typer.Argument(None, help="Optional session name to pre-fill")
MIC_DEVICE_OPTION = typer.Option(None, "--mic-device", help="Default microphone device id/name")
SYSTEM_DEVICE_OPTION = typer.Option(
    None, "--system-device", help="Default system audio device id/name"
)
MIX_DOWN_OPTION = typer.Option(True, "--mix-down/--no-mix-down", help="Create a mixed track")
TRANSCRIPTION_BACKEND_OPTION = typer.Option(
    BackendChoice.DUMMY,
    "--transcription-backend",
    case_sensitive=False,
    help="Transcription backend to pre-select: none/dummy/openai",
)
NOTES_BACKEND_OPTION = typer.Option(
    BackendChoice.DUMMY,
    "--notes-backend",
    case_sensitive=False,
    help="Notes backend to pre-select: none/dummy/openai",
)
SAMPLE_RATE_OPTION = typer.Option(None, help="Override sample rate")
CHANNELS_OPTION = typer.Option(None, help="Override number of channels")
INTERFACE_OPTION = typer.Option(
    InterfaceChoice.AUTO,
    "--interface",
    "-i",
    case_sensitive=False,
    help="User interface to launch: auto (default), console, or window.",
)

F = TypeVar("F", bound=Callable[..., Any])


def with_common_ui_options(command: F) -> F:
    def wrapper(
        name: Optional[str] = SESSION_NAME_ARGUMENT,
        mic_device: Optional[str] = MIC_DEVICE_OPTION,
        system_device: Optional[str] = SYSTEM_DEVICE_OPTION,
        mix_down: bool = MIX_DOWN_OPTION,
        transcription_backend: BackendChoice = TRANSCRIPTION_BACKEND_OPTION,
        notes_backend: BackendChoice = NOTES_BACKEND_OPTION,
        sample_rate: Optional[int] = SAMPLE_RATE_OPTION,
        channels: Optional[int] = CHANNELS_OPTION,
        interface: InterfaceChoice = INTERFACE_OPTION,
    ) -> Any:
        return command(
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

    wrapper.__name__ = getattr(command, "__name__", wrapper.__name__)
    wrapper.__qualname__ = getattr(command, "__qualname__", wrapper.__qualname__)
    wrapper.__doc__ = getattr(command, "__doc__", wrapper.__doc__)
    wrapper.__module__ = getattr(command, "__module__", wrapper.__module__)

    return cast(F, wrapper)


def _normalise_backend_choice(value: Union[BackendChoice, str]) -> BackendChoice:
    if isinstance(value, BackendChoice):
        return value
    try:
        return BackendChoice(value)
    except ValueError as exc:
        raise typer.BadParameter("Backend must be one of: none, dummy, openai") from exc


@app.callback()
def initialise_logging() -> None:
    """Initialise logging once per CLI invocation."""

    configure_logging()


def _launch_ui(
    *,
    name: Optional[str],
    mic_device: Optional[str],
    system_device: Optional[str],
    mix_down: bool,
    transcription_backend: Union[BackendChoice, str],
    notes_backend: Union[BackendChoice, str],
    sample_rate: Optional[int],
    channels: Optional[int],
    interface: InterfaceChoice,
) -> None:
    settings = get_settings()
    resolved_mic = mic_device if mic_device is not None else settings.default_mic_device
    resolved_system = (
        system_device if system_device is not None else settings.default_system_device
    )

    transcription_choice = _normalise_backend_choice(transcription_backend)
    notes_choice = _normalise_backend_choice(notes_backend)

    launch_config = UILaunchConfig(
        settings=settings,
        sample_rate=sample_rate,
        channels=channels,
        mix_down=mix_down,
        default_name=name,
        default_mic=resolved_mic,
        default_system=resolved_system,
        transcription_backend_name=transcription_choice,
        notes_backend_name=notes_choice,
    )

    ui_instance = _resolve_ui(interface, launch_config.to_kwargs())
    ui_instance.run()


def _resolve_ui(
    interface: Union[InterfaceChoice, str],
    config: Union[UILaunchConfig, Mapping[str, Any]],
):
    """Return the UI implementation based on the requested interface."""

    requested_value = interface.value if isinstance(interface, InterfaceChoice) else str(interface)

    if isinstance(interface, InterfaceChoice):
        normalized = interface
    else:
        try:
            normalized = InterfaceChoice(interface)
        except ValueError as exc:
            raise typer.BadParameter("Interface must be one of: auto, console, window") from exc

    if normalized not in {
        InterfaceChoice.AUTO,
        InterfaceChoice.CONSOLE,
        InterfaceChoice.WINDOW,
    }:
        raise typer.BadParameter("Interface must be one of: auto, console, window")

    if normalized is InterfaceChoice.AUTO:
        normalized = InterfaceChoice.WINDOW if sys.platform.startswith("win") else InterfaceChoice.CONSOLE

    if isinstance(config, UILaunchConfig):
        ui_kwargs = config.to_kwargs()
    elif isinstance(config, Mapping):
        ui_kwargs = dict(config)
    else:  # pragma: no cover - defensive fallback
        raise TypeError("UI configuration must be a mapping or UILaunchConfig instance")

    if normalized is InterfaceChoice.WINDOW:
        try:
            from .ui import RecordingWindowUI

            if RecordingWindowUI.is_supported():
                return RecordingWindowUI(**ui_kwargs)
            typer.secho(
                "Window UI is not available on this system. "
                f"Requested interface '{requested_value}'. "
                "Falling back to console interface.",
                err=True,
                fg="yellow",
            )
        except Exception as exc:  # pragma: no cover - runtime fallback
            typer.secho(
                f"Failed to initialise the window UI ({exc}). "
                f"Requested interface '{requested_value}'. "
                "Falling back to console.",
                err=True,
                fg="yellow",
            )

    return RecordingConsoleUI(**ui_kwargs)


@app.command()
def devices() -> None:
    """List available audio input devices."""

    typer.echo(format_device_table())


@app.command()
def desktop(
    port: int = typer.Option(0, help="Local server port. Use 0 to pick a free port."),
    browser: bool = typer.Option(False, "--browser", help="Open in the default browser instead of a desktop webview."),
) -> None:
    """Launch the ADsum v2 Windows desktop experience."""

    from .desktop import run_desktop_app

    run_desktop_app(port=port, browser=browser)


@app.command()
@with_common_ui_options
def ui(
    *,
    name: Optional[str],
    mic_device: Optional[str],
    system_device: Optional[str],
    mix_down: bool,
    transcription_backend: BackendChoice,
    notes_backend: BackendChoice,
    sample_rate: Optional[int],
    channels: Optional[int],
    interface: InterfaceChoice,
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
@with_common_ui_options
def record(
    *,
    name: Optional[str],
    mic_device: Optional[str],
    system_device: Optional[str],
    mix_down: bool,
    transcription_backend: BackendChoice,
    notes_backend: BackendChoice,
    sample_rate: Optional[int],
    channels: Optional[int],
    interface: InterfaceChoice,
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

