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
from ..core.audio.factory import (
    CaptureConfigurationError,
    CaptureRequest,
    DISABLED_DEVICE_SENTINEL,
    create_capture,
)
from ..core.audio.ffmpeg_backend import FFmpegBinaryNotFoundError, ensure_ffmpeg_available


DEVICE_DISABLE_KEYWORDS = {"skip", "none", "off", "disabled"}
from ..core.audio.devices import (
    FFmpegDeviceEnumerationError,
    format_device_table,
    format_ffmpeg_error_message,
)
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
        self._default_mic = self._normalize_device_value(
            default_mic if default_mic is not None else self._settings.default_mic_device
        )
        self._default_system = self._normalize_device_value(
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
        self._ffmpeg_prompted = False

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
        print(self._render_device_table())

        mic = self._normalize_device_value(self._prompt_device("Microphone", self._default_mic))
        system = self._normalize_device_value(
            self._prompt_device("System", self._default_system)
        )
        self._default_mic = mic
        self._default_system = system
        self._persist_device_setting("default_mic_device", mic, "microphone")
        self._persist_device_setting("default_system_device", system, "system")

        captures: Dict[str, AudioCapture] = {}
        for channel, device in {"microphone": mic, "system": system}.items():
            if device == DISABLED_DEVICE_SENTINEL:
                continue
            try:
                capture = create_capture(
                    CaptureRequest(
                        channel=channel,
                        device=device,
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        backend=self._settings.audio_backend,
                        chunk_seconds=self._settings.chunk_seconds,
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
        print(self._render_device_table())

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

    def _normalize_device_value(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value == DISABLED_DEVICE_SENTINEL:
            return DISABLED_DEVICE_SENTINEL
        stripped = value.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered in DEVICE_DISABLE_KEYWORDS:
            return DISABLED_DEVICE_SENTINEL
        if lowered in {"default", "auto"}:
            return None
        return stripped

    def _prompt_device(self, label: str, current: Optional[str]) -> Optional[str]:
        placeholder = self._format_device_display(current)
        prompt = (
            f"{label} device id/name [{placeholder}] "
            "(press Enter for default/current, type 'skip' to disable): "
        )
        value = input(prompt).strip()
        if not value:
            return current
        normalized = self._normalize_device_value(value)
        return normalized

    def _format_device_display(self, value: Optional[str]) -> str:
        if value == DISABLED_DEVICE_SENTINEL:
            return "disabled"
        if value:
            return value
        return "system default"

    def _render_device_table(self) -> str:
        try:
            return format_device_table()
        except FFmpegBinaryNotFoundError as exc:
            LOGGER.error("FFmpeg binary unavailable while listing devices: %s", exc)
            message = f"Unable to launch FFmpeg for device enumeration: {exc}"
            return format_ffmpeg_error_message(self._settings.ffmpeg_binary, message)
        except FFmpegDeviceEnumerationError as exc:
            LOGGER.error("FFmpeg device enumeration failed: %s", exc)
            message = f"Unable to enumerate FFmpeg audio devices: {exc}"
            return format_ffmpeg_error_message(self._settings.ffmpeg_binary, message)

    def _persist_device_setting(
        self, field: str, value: Optional[str], label: str
    ) -> None:
        value = self._normalize_device_value(value)
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
            self._default_mic = self._normalize_device_value(
                updated_settings.default_mic_device
            )
        elif field == "default_system_device":
            self._default_system = self._normalize_device_value(
                updated_settings.default_system_device
            )

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
        if value == DISABLED_DEVICE_SENTINEL:
            return "disabled"
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
                if not self._handle_recording_failure(self._pending_error):
                    self._error(f"Recording failed: {self._pending_error}")
            elif self._pending_outcome:
                self._last_outcome = self._pending_outcome
                summary = f"Session saved at {self._pending_outcome.session.id}"
                self._info(summary)
                if self._pending_outcome.channel_metrics:
                    for metrics in self._pending_outcome.channel_metrics.values():
                        if getattr(metrics, "is_silent", False):
                            device_label = metrics.device or "unknown device"
                            audio_path = self._pending_outcome.session.audio_paths.get(
                                metrics.channel
                            )
                            path_hint = f" (file: {audio_path})" if audio_path else ""
                            self._warning(
                                "No audio was captured from "
                                f"{metrics.channel} ({device_label}){path_hint}."
                            )
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

    def _handle_recording_failure(self, exc: Exception) -> bool:
        if isinstance(exc, FFmpegBinaryNotFoundError):
            self._error(str(exc))
            self._maybe_prompt_ffmpeg_path()
            return True
        return False

    def _maybe_prompt_ffmpeg_path(self) -> None:
        if self._ffmpeg_prompted:
            self._info(
                "FFmpeg path is still missing. Set ADSUM_FFMPEG_BINARY from the environment menu "
                "when you are ready."
            )
            return

        self._ffmpeg_prompted = True
        print()

        settings = self._settings
        download_url = settings.ffmpeg_download_url

        if download_url:
            choice = input(
                "Attempt to download FFmpeg automatically using ADSUM_FFMPEG_DOWNLOAD_URL? [y/N]: "
            ).strip().lower()
            if choice in {"y", "yes"}:
                self._info("Downloading FFmpeg. This may take a moment...")
                auto_path = ensure_ffmpeg_available(settings.ffmpeg_binary)
                if auto_path:
                    try:
                        updated = update_environment_setting("ffmpeg_binary", auto_path)
                    except EnvironmentSettingError as exc:
                        self._error(f"Failed to store FFmpeg path: {exc}")
                    else:
                        self._apply_settings(updated)
                        self._info(
                            f"FFmpeg downloaded to {auto_path} and saved to ADSUM_FFMPEG_BINARY."
                        )
                        return
                else:
                    self._error(
                        "Automatic FFmpeg download failed. Please choose the executable manually."
                    )
        else:
            remember = input(
                "Provide an FFmpeg download URL so ADsum can manage the binary automatically? [y/N]: "
            ).strip().lower()
            if remember in {"y", "yes"}:
                url_value = input(
                    "Enter the FFmpeg download URL (use {platform} as a placeholder when needed): "
                ).strip()
                if url_value:
                    try:
                        updated_settings = update_environment_setting("ffmpeg_download_url", url_value)
                    except EnvironmentSettingError as exc:
                        self._error(f"Failed to store FFmpeg download URL: {exc}")
                    else:
                        self._apply_settings(updated_settings)
                        self._info("Saved ADSUM_FFMPEG_DOWNLOAD_URL. Attempting download...")
                        auto_path = ensure_ffmpeg_available(
                            self._settings.ffmpeg_binary, download_url=url_value
                        )
                        if auto_path:
                            try:
                                updated_binary = update_environment_setting(
                                    "ffmpeg_binary", auto_path
                                )
                            except EnvironmentSettingError as exc:
                                self._error(f"Failed to store FFmpeg path: {exc}")
                            else:
                                self._apply_settings(updated_binary)
                                self._info(
                                    f"FFmpeg downloaded to {auto_path} and saved to ADSUM_FFMPEG_BINARY."
                                )
                                return
                        else:
                            self._error(
                                "Automatic FFmpeg download failed. Please choose the executable manually."
                            )

        choice = input(
            "Specify the full path to ffmpeg now? This will update ADSUM_FFMPEG_BINARY [y/N]: "
        ).strip().lower()
        if choice not in {"y", "yes"}:
            self._info(
                "FFmpeg binary not configured. Use the environment configuration menu to set "
                "ADSUM_FFMPEG_BINARY later."
            )
            return

        path = input("Enter the absolute path to ffmpeg (e.g. C:/Tools/ffmpeg/bin/ffmpeg.exe): ").strip()
        if not path:
            self._info(
                "No path provided. Re-run the command or open the environment menu to configure "
                "the FFmpeg binary later."
            )
            return

        try:
            updated = update_environment_setting("ffmpeg_binary", path)
        except EnvironmentSettingError as exc:
            self._error(f"Failed to store FFmpeg path: {exc}")
            return

        self._apply_settings(updated)
        self._info(f"FFmpeg path saved to ADSUM_FFMPEG_BINARY: {path}")

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

    def _warning(self, message: str) -> None:
        self._messages.append(f"[warning] {message}")

    def _error(self, message: str) -> None:
        self._messages.append(f"[error] {message}")

    def _flush_messages(self) -> None:
        while self._messages:
            print(self._messages.popleft())


__all__ = ["RecordingConsoleUI"]

