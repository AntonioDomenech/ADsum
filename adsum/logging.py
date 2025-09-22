"""Logging helpers for the ADsum project."""

from __future__ import annotations

import logging
from typing import Optional

_LOGGER_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic logging once for the application."""

    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    _LOGGER_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Convenience helper that ensures logging is configured."""

    configure_logging()
    return logging.getLogger(name or "adsum")


__all__ = ["configure_logging", "get_logger"]

