"""Structured JSON logger factory.

Each logger emits one JSON object per line to stdout, making logs easily
parseable by Databricks log ingestion, Splunk, or any structured log sink.

Usage::

    from src.logging.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started", extra={"run_id": run.info.run_id})
"""

from __future__ import annotations

import logging
import os


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter — no external dependency on python-json-logger."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via extra={}
        for key, val in record.__dict__.items():
            if key not in (
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "id",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            ):
                payload[key] = val
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a structured JSON logger for the given module name.

    Each logger is only configured once even when called multiple times.
    Log level defaults to INFO; override with the ``LOG_LEVEL`` env var.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger
