"""Shared utilities for console and window user interfaces."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from ..core.audio.factory import DISABLED_DEVICE_SENTINEL

# Common keywords that indicate a device should be disabled.
DEVICE_DISABLE_KEYWORDS = {"skip", "none", "off", "disabled"}


def normalize_device_value(value: Optional[str]) -> Optional[str]:
    """Return a normalised device string or ``None``/sentinel when disabled."""

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


def format_device_display(value: Optional[str]) -> str:
    """Return a user-facing label for a device selection."""

    if value == DISABLED_DEVICE_SENTINEL:
        return "disabled"
    if value:
        trimmed = value.strip().lower()
        if trimmed == "wasapi:default?loopback=1":
            return "system loopback (WASAPI)"
        return value
    return "system default"


def format_env_value(value: Any) -> str:
    """Format environment-backed values for display."""

    if value is None:
        return "(unset)"
    if value == DISABLED_DEVICE_SENTINEL:
        return "disabled"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def suggest_session_name() -> str:
    """Generate a timestamped default session name."""

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"Session {timestamp}"


__all__ = [
    "DEVICE_DISABLE_KEYWORDS",
    "format_device_display",
    "format_env_value",
    "normalize_device_value",
    "suggest_session_name",
]
