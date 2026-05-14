"""Lightweight logging configuration using rich for console output."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with Rich handler. Safe to call multiple times."""
    root = logging.getLogger()
    if any(isinstance(h, RichHandler) for h in root.handlers):
        return
    handler = RichHandler(rich_tracebacks=True, markup=False)
    formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the standard Rich-formatted handler."""
    configure_logging()
    return logging.getLogger(name)
