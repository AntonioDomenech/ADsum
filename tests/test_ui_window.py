"""Tests for the window UI helpers."""

from __future__ import annotations

from collections import deque

import pytest

from adsum.ui import window as window_module


def test_auto_detect_working_devices_skips_probe_for_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """When FFmpeg backend is active, probing should be skipped."""

    sentinel_message = "use ffmpeg instructions"

    def _fail_list_input_devices():
        raise AssertionError("sounddevice enumeration should not run for FFmpeg")

    def _capture_format_table(devices=None):  # type: ignore[override]
        assert devices is None
        return sentinel_message

    monkeypatch.setattr(window_module, "list_input_devices", _fail_list_input_devices)
    monkeypatch.setattr(window_module, "format_device_table", _capture_format_table)

    ui = object.__new__(window_module.RecordingWindowUI)
    ui._settings = type("_S", (), {"audio_backend": "ffmpeg"})()
    ui._messages = deque()
    ui._log_widget = None

    working, report = window_module.RecordingWindowUI._auto_detect_working_devices(ui)

    assert working == []
    assert report == sentinel_message
