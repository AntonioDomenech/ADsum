"""Graphical window UI for managing ADsum recordings."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Set

try:  # pragma: no cover - import guard for optional tkinter dependency
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk
    from tkinter.scrolledtext import ScrolledText
except Exception:  # pragma: no cover - headless environments may lack tkinter
    tk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    simpledialog = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    ScrolledText = None  # type: ignore[assignment]

from ..config import (
    EnvironmentSettingError,
    Settings,
    clear_environment_setting,
    get_settings,
    list_environment_settings,
    update_environment_setting,
)
from ..data.models import TranscriptResult
from ..core.audio.base import AudioCapture
from ..core.audio.devices import format_device_table
from ..core.audio.factory import CaptureConfigurationError, CaptureRequest, create_capture
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


class _UserCancelled(RuntimeError):
    """Raised when the user cancels an interactive dialog."""


class RecordingWindowUI:
    """Simple Tkinter-based UI for controlling recordings in a desktop window."""

    REFRESH_INTERVAL_MS = 500

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
        if tk is None:  # pragma: no cover - executed only when tkinter missing
            raise RuntimeError(
                "Tkinter is not available on this system. "
                "Install a Tk-enabled Python distribution to use the window UI."
            )

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
        self._transcription_backend_name = (transcription_backend_name or "none").lower()
        self._notes_backend_name = (notes_backend_name or "none").lower()

        self._orchestrator = RecordingOrchestrator()
        self._messages: Deque[str] = deque()
        self._transcript_queue: "queue.Queue[TranscriptResult]" = queue.Queue()
        self._transcript_results: Dict[str, TranscriptResult] = {}
        self._transcription_status: str = "Transcription results will appear here."
        self._active: Optional[_ActiveRecording] = None
        self._pending_outcome: Optional[RecordingOutcome] = None
        self._pending_error: Optional[Exception] = None
        self._last_outcome: Optional[RecordingOutcome] = None

        # Tk widgets set during run()
        self._root: Optional[tk.Tk] = None
        self._status_var: Optional[tk.StringVar] = None
        self._log_widget: Optional[ScrolledText] = None
        self._transcript_widget: Optional[ScrolledText] = None
        self._refresh_job: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - UI loop difficult to test automatically
        """Create the Tk window and enter the event loop."""

        assert tk is not None  # for type-checkers

        self._root = tk.Tk()
        self._root.title("ADsum Recorder")
        self._root.minsize(820, 640)
        self._root.geometry("920x720")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._status_var = tk.StringVar(value="No active recording.")

        self._configure_theme()

        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)

        main = ttk.Frame(self._root, padding=(24, 24, 24, 28), style="Main.TFrame")
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(4, weight=1)
        main.rowconfigure(5, weight=1)

        header = ttk.Label(main, text="ADsum Recorder", style="Header.TLabel")
        header.grid(row=0, column=0, sticky="w")

        subtitle = ttk.Label(
            main,
            text="Capture crystal-clear sessions with intuitive controls and live transcription.",
            style="Subheader.TLabel",
            wraplength=720,
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 20))

        status_card = ttk.Frame(main, padding=(16, 12), style="Card.TFrame")
        status_card.grid(row=2, column=0, sticky="ew", pady=(0, 20))
        status_card.columnconfigure(0, weight=1)

        ttk.Label(status_card, text="Session status", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Separator(status_card).grid(row=1, column=0, sticky="ew", pady=(8, 10))

        status_label = ttk.Label(status_card, textvariable=self._status_var, style="StatusValue.TLabel")
        status_label.grid(row=2, column=0, sticky="w")

        ttk.Label(
            status_card,
            text="Use Start to begin a session. Pause and Resume keep recording on your schedule.",
            style="Body.TLabel",
            wraplength=700,
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))

        controls_card = ttk.Frame(main, padding=(16, 12), style="Card.TFrame")
        controls_card.grid(row=3, column=0, sticky="ew", pady=(0, 20))
        controls_card.columnconfigure((0, 1, 2), weight=1)

        ttk.Label(controls_card, text="Quick controls", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Separator(controls_card).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 12))

        self._add_button(
            controls_card,
            "Start",
            self._start_recording,
            row=2,
            column=0,
            style="Accent.TButton",
        )
        self._add_button(
            controls_card,
            "Pause",
            self._pause_recording,
            row=2,
            column=1,
            style="Secondary.TButton",
        )
        self._add_button(
            controls_card,
            "Resume",
            self._resume_recording,
            row=2,
            column=2,
            style="Secondary.TButton",
        )
        self._add_button(
            controls_card,
            "Stop",
            self._stop_recording,
            row=3,
            column=0,
            style="Danger.TButton",
        )
        self._add_button(
            controls_card,
            "Test",
            self._test_devices,
            row=3,
            column=1,
            style="Tertiary.TButton",
        )
        self._add_button(
            controls_card,
            "Notes",
            self._show_notes,
            row=3,
            column=2,
            style="Tertiary.TButton",
        )
        self._add_button(
            controls_card,
            "Sessions",
            self._list_sessions,
            row=4,
            column=0,
            style="Tertiary.TButton",
        )
        self._add_button(
            controls_card,
            "Devices",
            self._show_devices,
            row=4,
            column=1,
            style="Tertiary.TButton",
        )
        self._add_button(
            controls_card,
            "Environment",
            self._configure_environment,
            row=4,
            column=2,
            style="Tertiary.TButton",
        )
        self._add_button(
            controls_card,
            "Quit",
            self._on_close,
            row=5,
            column=0,
            style="Danger.TButton",
            columnspan=3,
        )

        ttk.Label(
            controls_card,
            text="Tools give you fast access to previous sessions, devices and environment settings.",
            style="Caption.TLabel",
            wraplength=700,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 0))

        transcript_card = ttk.Frame(main, padding=(16, 12), style="Card.TFrame")
        transcript_card.grid(row=4, column=0, sticky="nsew", pady=(0, 20))
        transcript_card.columnconfigure(0, weight=1)
        transcript_card.rowconfigure(2, weight=1)

        ttk.Label(transcript_card, text="Live transcription", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Separator(transcript_card).grid(row=1, column=0, sticky="ew", pady=(8, 10))

        self._transcript_widget = ScrolledText(
            transcript_card,
            height=8,
            width=80,
            state="disabled",
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
        )
        self._transcript_widget.grid(row=2, column=0, sticky="nsew")
        self._transcript_widget.configure(
            font=("Segoe UI", 11),
            spacing1=4,
            spacing3=6,
            padx=12,
            pady=12,
            background="#ffffff",
            insertbackground="#0f172a",
        )

        log_card = ttk.Frame(main, padding=(16, 12), style="Card.TFrame")
        log_card.grid(row=5, column=0, sticky="nsew")
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(2, weight=1)

        ttk.Label(log_card, text="Activity log", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Separator(log_card).grid(row=1, column=0, sticky="ew", pady=(8, 10))

        self._log_widget = ScrolledText(
            log_card,
            height=12,
            width=80,
            state="disabled",
            wrap="word",
            borderwidth=0,
            highlightthickness=0,
        )
        self._log_widget.grid(row=2, column=0, sticky="nsew")
        self._log_widget.configure(
            font=("Segoe UI", 10),
            spacing1=2,
            spacing3=4,
            padx=12,
            pady=12,
            background="#ffffff",
            insertbackground="#0f172a",
        )

        self._info("Launching ADsum window UI. Close the window to exit.")
        self._flush_messages()
        self._render_transcription_text()
        self._schedule_refresh()

        self._root.mainloop()

    @classmethod
    def is_supported(cls) -> bool:
        """Return ``True`` when the Tk based UI can be used on this system."""

        return tk is not None

    # ------------------------------------------------------------------
    # UI plumbing
    # ------------------------------------------------------------------
    def _add_button(
        self,
        frame: "ttk.Frame",
        label: str,
        command,
        *,
        row: int,
        column: int,
        style: str,
        columnspan: int = 1,
    ) -> None:
        button = ttk.Button(frame, text=label, command=command, style=style)
        button.grid(
            row=row,
            column=column,
            columnspan=columnspan,
            sticky="ew",
            padx=6,
            pady=6,
        )

    def _schedule_refresh(self) -> None:
        if not self._root:
            return
        self._on_periodic_update()

    def _configure_theme(self) -> None:
        if not self._root:
            return

        base_bg = "#f5f7fa"
        card_bg = "#ffffff"
        accent = "#2563eb"
        accent_active = "#1d4ed8"
        accent_disabled = "#93c5fd"
        secondary_bg = "#e2e8f0"
        secondary_active = "#cbd5f5"
        tertiary_bg = "#f1f5f9"
        danger = "#ef4444"
        danger_active = "#dc2626"
        danger_disabled = "#fecaca"

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:  # pragma: no cover - depends on available Tk themes
            pass

        self._root.configure(background=base_bg)
        self._root.option_add("*Font", "Segoe UI 10")

        style.configure("Main.TFrame", background=base_bg)
        style.configure("Card.TFrame", background=card_bg, relief="flat", borderwidth=0)
        style.configure("Header.TLabel", background=base_bg, foreground="#0f172a", font=("Segoe UI", 22, "bold"))
        style.configure("Subheader.TLabel", background=base_bg, foreground="#475569", font=("Segoe UI", 11))
        style.configure("CardTitle.TLabel", background=card_bg, foreground="#0f172a", font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background=card_bg, foreground="#475569", font=("Segoe UI", 10))
        style.configure("Caption.TLabel", background=card_bg, foreground="#64748b", font=("Segoe UI", 9))
        style.configure(
            "StatusValue.TLabel",
            background=card_bg,
            foreground=accent,
            font=("Segoe UI", 12, "bold"),
        )

        style.configure("TButton", padding=(12, 10), font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TButton", relief=[("pressed", "sunken"), ("active", "raised")])

        style.configure("Accent.TButton", background=accent, foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("active", accent_active), ("disabled", accent_disabled)],
            foreground=[("disabled", "#e2e8f0")],
        )

        style.configure("Secondary.TButton", background=secondary_bg, foreground="#0f172a")
        style.map(
            "Secondary.TButton",
            background=[("active", secondary_active)],
            foreground=[("disabled", "#94a3b8")],
        )

        style.configure("Tertiary.TButton", background=tertiary_bg, foreground="#0f172a")
        style.map(
            "Tertiary.TButton",
            background=[("active", secondary_bg)],
            foreground=[("disabled", "#94a3b8")],
        )

        style.configure("Danger.TButton", background=danger, foreground="#ffffff")
        style.map(
            "Danger.TButton",
            background=[("active", danger_active), ("disabled", danger_disabled)],
            foreground=[("disabled", "#fee2e2")],
        )

    def _on_periodic_update(self) -> None:
        if not self._root or not self._root.winfo_exists():
            return

        self._refresh_state()
        self._flush_transcription_updates()
        self._update_status()
        self._flush_messages()

        self._refresh_job = self._root.after(self.REFRESH_INTERVAL_MS, self._on_periodic_update)

    def _on_close(self) -> None:
        self._info("Shutting down...")
        self._flush_messages()
        self._shutdown_active_recording()

        if self._root:
            if self._refresh_job is not None:
                self._root.after_cancel(self._refresh_job)
                self._refresh_job = None
            self._root.destroy()

    def _update_status(self) -> None:
        if not self._status_var:
            return
        status = "No active recording."
        if self._active and self._active.thread.is_alive():
            state = "paused" if self._active.control.is_paused else "recording"
            status = f"Active session: {self._active.request.name} ({state})"
        self._status_var.set(status)

    def _append_log(self, message: str) -> None:
        if not self._log_widget:
            return
        self._log_widget.configure(state="normal")
        self._log_widget.insert("end", message + "\n")
        self._log_widget.configure(state="disabled")
        self._log_widget.see("end")

    # ------------------------------------------------------------------
    # Transcription visualisation
    # ------------------------------------------------------------------
    def _render_transcription_text(self) -> None:
        if not self._transcript_widget:
            return

        self._transcript_widget.configure(state="normal")
        self._transcript_widget.delete("1.0", "end")

        if not self._transcript_results:
            placeholder = self._transcription_status or "Transcription results will appear here."
            self._transcript_widget.insert("1.0", placeholder)
        else:
            lines = []
            for channel in sorted(self._transcript_results):
                result = self._transcript_results[channel]
                lines.append(f"[{channel}]")
                if result.segments:
                    for segment in result.segments:
                        start = getattr(segment, "start", None)
                        end = getattr(segment, "end", None)
                        if start is not None and end is not None:
                            lines.append(f"  {start:6.2f}-{end:6.2f}s  {segment.text}")
                        else:
                            lines.append(f"  {segment.text}")
                text = result.text.strip()
                if text:
                    if result.segments:
                        lines.append("  Full text:")
                        lines.append(f"    {text}")
                    else:
                        lines.append(f"  {text}")
                lines.append("")
            content = "\n".join(lines).rstrip()
            self._transcript_widget.insert("1.0", content)

        self._transcript_widget.configure(state="disabled")
        self._transcript_widget.see("end")

    def _update_transcription_status(self, message: str) -> None:
        self._transcription_status = message
        if not self._transcript_results:
            self._render_transcription_text()

    def _reset_transcription_view(self, status: Optional[str] = None) -> None:
        self._transcript_results.clear()
        self._clear_transcript_queue()
        self._transcription_status = status or "Transcription results will appear here."
        self._render_transcription_text()

    def _clear_transcript_queue(self) -> None:
        while True:
            try:
                self._transcript_queue.get_nowait()
            except queue.Empty:
                break

    def _flush_transcription_updates(self) -> None:
        updated_channels: Set[str] = set()
        while True:
            try:
                result = self._transcript_queue.get_nowait()
            except queue.Empty:
                break
            self._transcript_results[result.channel] = result
            updated_channels.add(result.channel)

        if updated_channels:
            self._transcription_status = ""
            self._render_transcription_text()

    def _on_transcript_result(self, result: TranscriptResult) -> None:
        self._transcript_queue.put(result)

    # ------------------------------------------------------------------
    # Menu handlers
    # ------------------------------------------------------------------
    def _start_recording(self) -> None:
        if self._active and self._active.thread.is_alive():
            self._info("A recording is already in progress. Stop it before starting a new one.")
            return

        try:
            name_default = self._default_name or self._suggest_session_name()
            name = self._prompt_session_name(name_default)
            self._default_name = name

            self._show_text_window("Available audio devices", format_device_table())

            mic = self._prompt_device("Microphone", self._default_mic)
            system = self._prompt_device("System", self._default_system)
            self._default_mic = mic
            self._default_system = system
            self._persist_device_setting("default_mic_device", mic, "microphone")
            self._persist_device_setting("default_system_device", system, "system")

            captures: Dict[str, AudioCapture] = {}
            for channel, device in {"microphone": mic, "system": system}.items():
                if device is None:
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

            mix_down = self._prompt_mix_down(self.mix_down)
            self.mix_down = mix_down

            transcription_name = (
                self._prompt_backend("Transcription backend", self._transcription_backend_name) or "none"
            ).strip().lower()
            notes_name = (
                self._prompt_backend("Notes backend", self._notes_backend_name) or "none"
            ).strip().lower()

            try:
                transcription = resolve_transcription_backend(transcription_name)
            except ServiceConfigurationError as exc:
                self._error(str(exc))
                return
            except Exception as exc:  # pragma: no cover - runtime errors handled at runtime
                LOGGER.exception("Failed to initialise transcription backend: %s", exc)
                self._error(f"Failed to initialise transcription backend: {exc}")
                return

            try:
                notes = resolve_notes_backend(notes_name)
            except ServiceConfigurationError as exc:
                self._error(str(exc))
                return
            except Exception as exc:  # pragma: no cover - runtime errors handled at runtime
                LOGGER.exception("Failed to initialise notes backend: %s", exc)
                self._error(f"Failed to initialise notes backend: {exc}")
                return

            self._transcription_backend_name = transcription_name
            self._notes_backend_name = notes_name

            if transcription is None:
                self._reset_transcription_view("Transcription disabled for this session.")
            else:
                self._reset_transcription_view("Recording... awaiting transcription results.")

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
            self._info("Recording started. Use pause/stop buttons to control the session.")

        except _UserCancelled:
            self._info("Recording setup cancelled.")

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

    def _test_devices(self) -> None:
        if not self._root:
            return

        self._show_text_window("Available audio devices", format_device_table())

        last_value = self._default_mic or self._default_system or ""
        tested_any = False

        while True:
            try:
                selection = self._prompt_test_device(last_value)
            except _UserCancelled:
                if tested_any:
                    self._info("Device testing cancelled.")
                else:
                    self._info("Device testing dismissed.")
                return

            if selection is None:
                if tested_any:
                    self._info("Finished testing audio devices.")
                else:
                    self._info("No device selected for testing.")
                return

            tested_any = True
            last_value = selection
            self._perform_device_test(selection)

    def _show_notes(self) -> None:
        if self._last_outcome and self._last_outcome.notes:
            notes = self._last_outcome.notes
            action_items = "\n".join(f"  {idx}. {item}" for idx, item in enumerate(notes.action_items, start=1))
            if not action_items:
                action_items = "No action items recorded."
            content = (
                f"Title: {notes.title}\n\n"
                f"Summary:\n{notes.summary}\n\n"
                f"Action items:\n{action_items}"
            )
            self._show_text_window("Latest notes", content)
        else:
            self._info("No notes available yet. Complete a recording with notes enabled.")

    def _list_sessions(self) -> None:
        sessions = self._orchestrator.store.list_sessions(limit=10)
        if not sessions:
            self._info("No sessions stored yet.")
            return

        lines = ["Recent sessions:"]
        for session in sessions:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.created_at))
            duration = f"{session.duration:.1f}s"
            lines.append(f"- {session.id} | {timestamp} | {session.name} ({duration})")
        self._show_text_window("Stored sessions", "\n".join(lines))

    def _show_devices(self) -> None:
        self._show_text_window("Audio devices", format_device_table())

    def _configure_environment(self) -> None:
        settings_entries = list(list_environment_settings(self._settings))
        if not settings_entries:
            self._info("No configurable environment variables detected.")
            return

        assert tk is not None and ttk is not None

        window = tk.Toplevel(self._root)
        window.title("Environment configuration")
        window.transient(self._root)
        window.grab_set()

        list_var = tk.StringVar(value=[self._format_setting_entry(entry) for entry in settings_entries])

        listbox = tk.Listbox(window, listvariable=list_var, height=min(len(settings_entries), 12), width=80)
        listbox.grid(row=0, column=0, columnspan=3, padx=12, pady=(12, 6), sticky="nsew")

        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)

        def refresh_entries() -> None:
            nonlocal settings_entries
            settings_entries = list(list_environment_settings(self._settings))
            list_var.set([self._format_setting_entry(entry) for entry in settings_entries])

        def require_selection() -> Optional[int]:
            selection = listbox.curselection()
            if not selection:
                messagebox.showinfo("Environment", "Select a variable first.", parent=window)
                return None
            return selection[0]

        def update_selected() -> None:
            index = require_selection()
            if index is None:
                return
            entry = settings_entries[index]
            current_value = "" if entry.value is None else str(entry.value)
            prompt = (
                f"Enter new value for {entry.env_name}.\n"
                "Leave empty to cancel."
            )
            new_value = simpledialog.askstring(
                "Update environment variable",
                prompt,
                parent=window,
                initialvalue=current_value,
            )
            if new_value is None or not new_value.strip():
                return
            try:
                updated_settings = update_environment_setting(entry.field, new_value)
            except EnvironmentSettingError as exc:
                self._error(f"Failed to update {entry.env_name}: {exc}")
                return
            self._apply_settings(updated_settings)
            current = getattr(self._settings, entry.field)
            self._info(
                f"{entry.env_name} updated. Current value: {self._format_env_value(current)}."
            )
            refresh_entries()

        def reset_selected() -> None:
            index = require_selection()
            if index is None:
                return
            entry = settings_entries[index]
            if not messagebox.askyesno(
                "Reset environment variable",
                f"Reset {entry.env_name} to its default value?",
                parent=window,
            ):
                return
            try:
                updated_settings = clear_environment_setting(entry.field)
            except EnvironmentSettingError as exc:
                self._error(f"Failed to reset {entry.env_name}: {exc}")
                return
            self._apply_settings(updated_settings)
            current = getattr(self._settings, entry.field)
            self._info(
                f"{entry.env_name} reset. Current value: {self._format_env_value(current)}."
            )
            refresh_entries()

        update_button = ttk.Button(window, text="Update", command=update_selected)
        update_button.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="w")

        reset_button = ttk.Button(window, text="Reset", command=reset_selected)
        reset_button.grid(row=1, column=1, padx=12, pady=(0, 12))

        close_button = ttk.Button(window, text="Close", command=window.destroy)
        close_button.grid(row=1, column=2, padx=12, pady=(0, 12), sticky="e")

        window.wait_window()

    # ------------------------------------------------------------------
    # Dialog helpers
    # ------------------------------------------------------------------
    def _prompt_session_name(self, default: str) -> str:
        assert simpledialog is not None
        name = simpledialog.askstring(
            "Session name",
            "Enter a name for this recording session:",
            initialvalue=default,
            parent=self._root,
        )
        if name is None:
            raise _UserCancelled
        name = name.strip()
        return name or default

    def _prompt_test_device(self, current: Optional[str]) -> Optional[str]:
        assert simpledialog is not None
        prompt = (
            "Enter the device id or name to test.\n"
            "Leave empty to finish testing."
        )
        value = simpledialog.askstring(
            "Test audio device",
            prompt,
            initialvalue=current or "",
            parent=self._root,
        )
        if value is None:
            raise _UserCancelled
        value = value.strip()
        if not value:
            return None
        return value

    def _prompt_device(self, label: str, current: Optional[str]) -> Optional[str]:
        assert simpledialog is not None
        current_value = current or ""
        current_display = self._format_device_display(current)
        value = simpledialog.askstring(
            f"{label} device",
            (
                f"Enter the {label.lower()} device id/name.\n"
                f"Current selection: {current_display}.\n"
                "Leave empty to keep the current value or type 'skip' to disable."
            ),
            initialvalue=current_value,
            parent=self._root,
        )
        if value is None:
            raise _UserCancelled
        value = value.strip()
        if not value:
            return current
        if value.lower() in {"skip", "none", "off", "disabled"}:
            return None
        return value

    def _prompt_mix_down(self, current: bool) -> bool:
        assert messagebox is not None
        result = messagebox.askyesnocancel(
            "Create mixed track",
            (
                "Create a mixed audio track combining all inputs?\n"
                f"Select 'Cancel' to keep the current setting ({'enabled' if current else 'disabled'})."
            ),
            parent=self._root,
        )
        if result is None:
            return current
        return bool(result)

    def _prompt_backend(self, label: str, current: Optional[str]) -> str:
        assert simpledialog is not None
        current = current or "none"
        value = simpledialog.askstring(
            label,
            f"Select {label.lower()} (none/dummy/openai):",
            initialvalue=current,
            parent=self._root,
        )
        if value is None:
            raise _UserCancelled
        value = value.strip()
        return value or current

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

    def _perform_device_test(self, device: str) -> None:
        request = CaptureRequest(
            channel="test",
            device=device,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

        try:
            capture = create_capture(request)
        except CaptureConfigurationError as exc:
            self._error(f"Failed to configure device {device}: {exc}")
            return

        if capture is None:
            self._error(
                f"No capture backend available for device {device}. "
                "Enter a numeric id or device name."
            )
            return

        try:
            capture.start()
        except Exception as exc:  # pragma: no cover - depends on runtime backend
            LOGGER.exception("Failed to start capture for device %s: %s", device, exc)
            self._error(f"Failed to start capture for {device}: {exc}")
            with contextlib.suppress(Exception):
                capture.close()
            return

        try:
            chunk = None
            deadline = time.time() + 3.0
            while time.time() < deadline:
                chunk = capture.read(timeout=0.3)
                if chunk is not None and getattr(chunk, "size", 0) > 0:
                    break

            if chunk is None or getattr(chunk, "size", 0) == 0:
                self._error(
                    f"No audio data received from {device}. "
                    "Check the device and ensure it is not muted or busy."
                )
                return

            shape = getattr(chunk, "shape", None)
            message = (
                f"Successfully captured audio from {device} "
                f"({capture.info.sample_rate} Hz, {capture.info.channels} channel(s))."
            )
            if shape:
                message += f" Sample shape: {shape}."
            self._info(message)
            if messagebox is not None and self._root is not None:
                messagebox.showinfo("Device test", message, parent=self._root)

        except Exception as exc:  # pragma: no cover - depends on runtime backend
            LOGGER.exception("Error while testing device %s: %s", device, exc)
            self._error(f"Error while testing {device}: {exc}")
        finally:
            with contextlib.suppress(Exception):
                capture.stop()
            with contextlib.suppress(Exception):
                capture.close()

    def _show_text_window(self, title: str, content: str) -> None:
        if not self._root:
            return
        window = tk.Toplevel(self._root)
        window.title(title)
        window.transient(self._root)
        window.grab_set()

        text = ScrolledText(window, width=90, height=24, state="normal")
        text.insert("1.0", content)
        text.configure(state="disabled")
        text.pack(fill="both", expand=True, padx=12, pady=12)

        button = ttk.Button(window, text="Close", command=window.destroy)
        button.pack(pady=(0, 12))

        window.wait_window()

    # ------------------------------------------------------------------
    # Internal helpers shared with console UI
    # ------------------------------------------------------------------
    def _ensure_active(self) -> Optional[_ActiveRecording]:
        if not self._active or not self._active.thread.is_alive():
            self._info("No active recording.")
            return None
        return self._active

    def _suggest_session_name(self) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return f"Session {timestamp}"

    def _apply_settings(self, settings: Settings) -> None:
        self._settings = settings
        if self._cli_sample_rate is None:
            self.sample_rate = settings.sample_rate
        if self._cli_channels is None:
            self.channels = settings.channels

    def _format_setting_entry(self, entry) -> str:
        return (
            f"{entry.env_name} = {self._format_env_value(entry.value)} "
            f"(default: {self._format_env_value(entry.default)})"
        )

    def _format_env_value(self, value: Any) -> str:
        if value is None:
            return "(unset)"
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
                transcript_callback=self._on_transcript_result,
                transcript_update_callback=self._on_transcript_result,
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
                if not self._transcript_results:
                    self._update_transcription_status(
                        "Recording failed before transcription could complete."
                    )
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
                    for result in self._pending_outcome.transcripts.values():
                        self._transcript_results[result.channel] = result
                    self._transcription_status = ""
                    self._render_transcription_text()
                elif not self._transcript_results and (
                    "awaiting transcription" in self._transcription_status.lower()
                    or "will appear here" in self._transcription_status.lower()
                ):
                    self._update_transcription_status(
                        "No transcription available for this session."
                    )
                if self._pending_outcome.notes:
                    self._info("Notes summary available via the Notes button.")
            self._pending_error = None
            self._pending_outcome = None

    def _shutdown_active_recording(self) -> None:
        if self._active and self._active.thread.is_alive():
            self._active.control.request_stop()
            self._active.thread.join()
            self._active = None

    def _info(self, message: str) -> None:
        self._messages.append(f"[info] {message}")
        self._append_log(f"[info] {message}")

    def _error(self, message: str) -> None:
        self._messages.append(f"[error] {message}")
        self._append_log(f"[error] {message}")
        if messagebox is not None and self._root is not None:
            messagebox.showerror("ADsum", message, parent=self._root)

    def _flush_messages(self) -> None:
        while self._messages:
            message = self._messages.popleft()
            self._append_log(message)


__all__ = ["RecordingWindowUI"]

