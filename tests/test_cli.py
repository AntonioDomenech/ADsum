"""Tests for CLI helpers."""

from __future__ import annotations

import types

import pytest
import typer

from adsum import cli
from adsum.config import get_settings


def _build_kwargs() -> dict:
    settings = get_settings()
    return {
        "settings": settings,
        "sample_rate": None,
        "channels": None,
        "mix_down": True,
        "default_name": None,
        "default_mic": settings.default_mic_device,
        "default_system": settings.default_system_device,
        "transcription_backend_name": "dummy",
        "notes_backend_name": "dummy",
    }


def test_resolve_ui_console() -> None:
    kwargs = _build_kwargs()
    ui = cli._resolve_ui("console", kwargs)
    assert isinstance(ui, cli.RecordingConsoleUI)


def test_resolve_ui_invalid_option() -> None:
    kwargs = _build_kwargs()
    with pytest.raises(typer.BadParameter):
        cli._resolve_ui("invalid", kwargs)


class _DummyWindowUI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self) -> None:  # pragma: no cover - not exercised
        pass

    @classmethod
    def is_supported(cls) -> bool:
        return True


def test_resolve_ui_auto_prefers_window(monkeypatch) -> None:
    kwargs = _build_kwargs()
    monkeypatch.setattr(cli, "sys", types.SimpleNamespace(platform="win32"))

    import adsum.ui as ui_module

    monkeypatch.setattr(ui_module, "RecordingWindowUI", _DummyWindowUI)

    ui = cli._resolve_ui("auto", kwargs)
    assert isinstance(ui, _DummyWindowUI)
    assert ui.kwargs == kwargs


class _UnsupportedWindowUI(_DummyWindowUI):
    @classmethod
    def is_supported(cls) -> bool:
        return False


def test_resolve_ui_window_falls_back(monkeypatch) -> None:
    kwargs = _build_kwargs()
    monkeypatch.setattr(cli, "sys", types.SimpleNamespace(platform="win32"))

    import adsum.ui as ui_module

    monkeypatch.setattr(ui_module, "RecordingWindowUI", _UnsupportedWindowUI)

    ui = cli._resolve_ui("window", kwargs)
    assert isinstance(ui, cli.RecordingConsoleUI)


def test_launch_ui_uses_settings_defaults(monkeypatch) -> None:
    captured: dict = {}

    def fake_configure_logging() -> None:
        captured["logging"] = True

    class _DummyUI:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run(self) -> None:
            captured["ran"] = True

    def fake_resolve(interface: str, kwargs: dict):
        captured["interface"] = interface
        return _DummyUI(**kwargs)

    base_settings = get_settings()
    custom_settings = base_settings.model_copy(
        update={"default_mic_device": "MicX", "default_system_device": "SysY"}
    )

    monkeypatch.setattr(cli, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(cli, "_resolve_ui", fake_resolve)
    monkeypatch.setattr(cli, "get_settings", lambda: custom_settings)

    cli._launch_ui(
        name=None,
        mic_device=None,
        system_device=None,
        mix_down=True,
        transcription_backend="dummy",
        notes_backend="dummy",
        sample_rate=None,
        channels=None,
        interface="console",
    )

    kwargs = captured["kwargs"]
    assert captured["interface"] == "console"
    assert kwargs["settings"] is custom_settings
    assert kwargs["default_mic"] == "MicX"
    assert kwargs["default_system"] == "SysY"
    assert captured.get("ran") is True

