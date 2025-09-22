"""Tests for audio device utilities."""

from __future__ import annotations

from adsum.core.audio import devices


def test_format_device_table_fallback_contains_install_hint(monkeypatch) -> None:
    """When no devices are found the fallback message should include install hint."""

    monkeypatch.setattr(devices, "list_input_devices", lambda: [])

    message = devices.format_device_table()

    assert message.startswith("No input devices detected.")
    assert "pip install adsum[audio]" in message
