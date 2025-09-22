"""Interactive console UI for managing ADsum recordings."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from ..config import (
    EnvironmentSettingError,
    Settings,
    clear_environment_setting,
    get_settings,
    list_environment_settings,
    update_environment_setting,
)
from ..core.audio.base import AudioCapture
from ..core.audio.factory import CaptureConfigurationError, CaptureRequest, create_capture
from ..core.audio.devices import format_device_table
from ..core.pipeline.orchestrator import (
    RecordingControl,
    RecordingOrchestrator,
    RecordingOutcome,
    RecordingRequest,
)
from ..logging import get_logger
from ..services.factory import (
    ServiceConfigurationError,
    resolve_notes_backend,
    resolve_transcription_backend,
)

LOGGER = get_logger(__name__)


@dataclass
class _ActiveRecording:
    request: RecordingRequest
    control: RecordingControl
    thread: threading.Thread


class RecordingConsoleUI:
    """Simple interactive console used to manage recordings from the terminal."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        mix_down: bool,
        default_name: Optional[str] = None,
        default_mic: Optional[str] = None,
        default_system: Optional[str] = None,
        transcription_backend_name: Optional[str] = "dummy",
        notes_backend_name: Optional[str] = "dummy",
    ) -> None:
        self._settings = settings or get_settings()
        self._cli_sample_rate = sample_rate
        self._cli_channels = channels
        self.sample_rate = sample_rate if sample_rate is not None else self._settings.sample_rate
        self.channels = channels if channels is not None else self._settings.channels
        self.mix_down = mix_down
        self._default_name = default_name
        self._default_mic = (
            default_mic if default_mic is not None else self._settings.default_mic_device
        )
        self._default_system = (
            default_system
            if default_system is not None
            else self._settings.default_system_device
        )
        self._transcription_backend_name = transcription_backend_name or "none"
        self._notes_backend_name = notes_backend_name or "none"

        self._orchestrator = RecordingOrchestrator()
        self._messages: Deque[str] = deque()
        self._active: Optional[_ActiveRecording] = None
        self._pending_outcome: Optional[RecordingOutcome] = None
        self._pending_error: Optional[Exception] = None
        self._last_outcome: Optional[RecordingOutcome] = None
        self._running = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Enter the interactive UI loop."""

        self._info("Launching ADsum interactive UI. Press Ctrl+C to exit.")
        try:
            while self._running:
                self._refresh_state()
                self._flush_messages()
                self._print_menu()
                try:
                    choice = input("Select option: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print()
                    choice = "q"
                self._handle_choice(choice)
        finally:
            self._shutdown_active_recording()
            self._flush_messages()
            print("Goodbye!")

    # ------------------------------------------------------------------
    # Menu handlers
    # ------------------------------------------------------------------
    def _handle_choice(self, choice: str) -> None:
        if choice in {"1", "start", "s"}:
            self._start_recording()
        elif choice in {"2", "pause", "p"}:
            self._pause_recording()
        elif choice in {"3", "resume", "r"}:
            self._resume_recording()
        elif choice in {"4", "stop", "x"}:
            self._stop_recording()
        elif choice in {"5", "notes", "n"}:
            self._show_notes()
        elif choice in {"6", "sessions", "list", "l"}:
            self._list_sessions()
        elif choice in {"7", "devices", "d"}:
            self._show_devices()
        elif choice in {"8", "env", "config", "e"}:
            self._configure_environment()
        elif choice in {"q", "quit", "exit"}:
            self._running = False
        else:
            self._info("Unknown option. Please choose one of the menu entries.")

    def _start_recording(self) -> None:
        if self._active and self._active.thread.is_alive():
            self._info("A recording is already in progress. Stop it before starting a new one.")
            return

        name_default = self._default_name or self._suggest_session_name()
        name = input(f"Session name [{name_default}]: ").strip() or name_default
        self._default_name = name

        print()
        print(format_device_table())

        mic = self._prompt_device("Microphone", self._default_mic)
        system = self._prompt_device("System", self._default_system)
        self._default_mic = mic
        self._default_system = system
        self._persist_device_setting("default_mic_device", mic, "microphone")
        self._persist_device_setting("default_system_device", system, "system")

        captures: Dict[str, AudioCapture] = {}
        for channel, device in {"microphone": mic, "system": system}.items():
            try:
                capture = create_capture(
                    CaptureRequest(
                        channel=channel,
                        device=device,
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                    )
                )
            except CaptureConfigurationError as exc:
                self._error(f"Failed to configure {channel} capture: {exc}")
                continue
            if capture is not None:
                captures[channel] = capture

        if not captures:
            self._error("No audio devices configured. Recording aborted.")
            return

        mix_down = self._prompt_bool("Create mixed track", self.mix_down)
        self.mix_down = mix_down

        transcription_name = (
            self._prompt_backend("Transcription backend", self._transcription_backend_name)
            or "none"
        ).strip().lower()
        notes_name = (
            self._prompt_backend("Notes backend", self._notes_backend_name) or "none"
        ).strip().lower()

        try:
            transcription = resolve_transcription_backend(transcription_name)
        except ServiceConfigurationError as exc:
            self._error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime errors handled interactively
            LOGGER.exception("Failed to initialise transcription backend: %s", exc)
            self._error(f"Failed to initialise transcription backend: {exc}")
            return

        try:
            notes = resolve_notes_backend(notes_name)
        except ServiceConfigurationError as exc:
            self._error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime errors handled interactively
            LOGGER.exception("Failed to initialise notes backend: %s", exc)
            self._error(f"Failed to initialise notes backend: {exc}")
            return

        self._transcription_backend_name = transcription_name
        self._notes_backend_name = notes_name

        request = RecordingRequest(name=name, captures=captures, mix_down=mix_down)
        control = RecordingControl()

        thread = threading.Thread(
            target=self._run_recording,
            args=(request, control, transcription, notes),
            daemon=True,
        )

        self._pending_outcome = None
        self._pending_error = None
        self._active = _ActiveRecording(request=request, control=control, thread=thread)

        thread.start()
        self._info("Recording started. Use pause/stop options to control the session.")

    def _pause_recording(self) -> None:
        active = self._ensure_active()
        if not active:
            return
        if active.control.is_paused:
            self._info("Recording is already paused.")
            return
        active.control.request_pause()
        self._info("Recording paused.")

    def _resume_recording(self) -> None:
        active = self._ensure_active()
        if not active:
            return
        if not active.control.is_paused:
            self._info("Recording is not currently paused.")
            return
        active.control.request_resume()
        self._info("Recording resumed.")

    def _stop_recording(self) -> None:
        active = self._ensure_active()
        if not active:
            return
        active.control.request_stop()
        self._info("Stopping recording...")

    def _show_notes(self) -> None:
        if self._last_outcome and self._last_outcome.notes:
            notes = self._last_outcome.notes
            print()
            print(f"Title: {notes.title}")
            print(f"Summary: {notes.summary}")
            if notes.action_items:
                print("Action items:")
                for idx, item in enumerate(notes.action_items, start=1):
                    print(f"  {idx}. {item}")
            else:
                print("No action items recorded.")
        else:
            self._info("No notes available yet. Complete a recording with notes enabled.")

    def _list_sessions(self) -> None:
        sessions = self._orchestrator.store.list_sessions(limit=10)
        if not sessions:
            self._info("No sessions stored yet.")
            return
        print()
        print("Recent sessions:")
        for session in sessions:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.created_at))
            duration = f"{session.duration:.1f}s"
            print(f"- {session.id} | {timestamp} | {session.name} ({duration})")

    def _show_devices(self) -> None:
        print()
        print(format_device_table())

    def _configure_environment(self) -> None:
        while True:
            settings = list(list_environment_settings(self._settings))
            if not settings:
                self._info("No configurable environment variables detected.")
                return

            print()
            print("Environment configuration:")
            for idx, entry in enumerate(settings, start=1):
                print(
                    f"{idx}) {entry.env_name} = {self._format_env_value(entry.value)}"
                    f" (default: {self._format_env_value(entry.default)})"
                )
            print("b) Back to main menu")

            choice = input("Select variable to edit: ").strip().lower()
            if choice in {"b", "back", "q", "exit"}:
                return

            try:
                index = int(choice)
            except ValueError:
                self._info("Invalid selection. Choose a number from the list or 'b' to go back.")
                continue

            if not 1 <= index <= len(settings):
                self._info("Selection out of range. Try again.")
                continue

            selected = settings[index - 1]
            prompt = (
                f"Enter new value for {selected.env_name}"
                " (leave empty to reset to default): "
            )
            new_value = input(prompt).strip()

            try:
                if new_value:
                    updated_settings = update_environment_setting(selected.field, new_value)
                    verb = "updated"
                else:
                    updated_settings = clear_environment_setting(selected.field)
                    verb = "reset"
            except EnvironmentSettingError as exc:
                self._error(f"Failed to update {selected.env_name}: {exc}")
                continue

            self._apply_settings(updated_settings)
            current = getattr(self._settings, selected.field)
            self._info(
                f"{selected.env_name} {verb}. Current value: {self._format_env_value(current)}."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _suggest_session_name(self) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return f"Session {timestamp}"

    def _prompt_device(self, label: str, current: Optional[str]) -> Optional[str]:
        placeholder = self._format_device_display(current)
        prompt = (
            f"{label} device id/name [{placeholder}] "
            "(press Enter to keep, type 'skip' to disable): "
        )
        value = input(prompt).strip()
        if not value:
            return current
        lowered = value.lower()
        if lowered in {"skip", "none", "off", "disabled"}:
            return None
        return value

    def _format_device_display(self, value: Optional[str]) -> str:
        if value:
            return value
        return "disabled"

    def _persist_device_setting(
        self, field: str, value: Optional[str], label: str
    ) -> None:
        try:
            if value is None:
                updated_settings = clear_environment_setting(field)
            else:
                updated_settings = update_environment_setting(field, value)
        except EnvironmentSettingError as exc:
            self._error(f"Failed to store default {label} device: {exc}")
            return

        self._apply_settings(updated_settings)
        if field == "default_mic_device":
            self._default_mic = updated_settings.default_mic_device
        elif field == "default_system_device":
            self._default_system = updated_settings.default_system_device

    def _prompt_bool(self, label: str, current: bool) -> bool:
        suffix = "Y/n" if current else "y/N"
        value = input(f"{label}? ({suffix}): ").strip().lower()
        if not value:
            return current
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        self._info("Invalid response. Keeping previous value.")
        return current

    def _prompt_backend(self, label: str, current: Optional[str]) -> str:
        current = current or "none"
        value = input(f"{label} [{current}]: ").strip()
        return value or current

    def _ensure_active(self) -> Optional[_ActiveRecording]:
        if not self._active or not self._active.thread.is_alive():
            self._info("No active recording.")
            return None
        return self._active

    def _apply_settings(self, settings: Settings) -> None:
        self._settings = settings
        if self._cli_sample_rate is None:
            self.sample_rate = settings.sample_rate
        if self._cli_channels is None:
            self.channels = settings.channels

    def _format_env_value(self, value: Any) -> str:
        if value is None:
            return "(unset)"
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _run_recording(
        self,
        request: RecordingRequest,
        control: RecordingControl,
        transcription,
        notes,
    ) -> None:
        try:
            outcome = self._orchestrator.record(
                request,
                duration=None,
                transcription=transcription,
                notes=notes,
                control=control,
            )
            self._pending_outcome = outcome
        except Exception as exc:  # pragma: no cover - runtime behaviour
            LOGGER.exception("Recording thread failed: %s", exc)
            self._pending_error = exc

    def _refresh_state(self) -> None:
        if self._active and not self._active.thread.is_alive():
            self._active.thread.join()
            self._active = None
            if self._pending_error:
                self._error(f"Recording failed: {self._pending_error}")
            elif self._pending_outcome:
                self._last_outcome = self._pending_outcome
                summary = f"Session saved at {self._pending_outcome.session.id}"
                self._info(summary)
                if self._pending_outcome.transcripts:
                    counts = ", ".join(
                        f"{ch}: {len(result.text.split())} words"
                        for ch, result in self._pending_outcome.transcripts.items()
                    )
                    self._info(f"Transcripts generated ({counts}).")
                if self._pending_outcome.notes:
                    self._info("Notes summary available via the notes menu option.")
            self._pending_error = None
            self._pending_outcome = None

    def _shutdown_active_recording(self) -> None:
        if self._active and self._active.thread.is_alive():
            self._active.control.request_stop()
            self._active.thread.join()
            self._active = None

    def _print_menu(self) -> None:
        print()
        status = "No active recording."
        if self._active and self._active.thread.is_alive():
            state = "paused" if self._active.control.is_paused else "recording"
            status = f"Active session: {self._active.request.name} ({state})"
        print(status)
        print("1) Start recording")
        print("2) Pause recording")
        print("3) Resume recording")
        print("4) Stop recording")
        print("5) Show latest notes")
        print("6) List stored sessions")
        print("7) Show audio devices")
        print("8) Configure environment variables")
        print("q) Quit")

    def _info(self, message: str) -> None:
        self._messages.append(f"[info] {message}")

    def _error(self, message: str) -> None:
        self._messages.append(f"[error] {message}")

    def _flush_messages(self) -> None:
        while self._messages:
            print(self._messages.popleft())


__all__ = ["RecordingConsoleUI"]

