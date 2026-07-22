"""Readable structured logging helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar


class JsonFormatter(logging.Formatter):
    """Small JSON-lines formatter suitable for local diagnostics."""

    _reserved: ClassVar[set[str]] = set(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._reserved and key not in {"message", "asctime"}:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "INFO",
    log_dir: Path = Path("data/logs"),
    *,
    max_bytes: int = 10_000_000,
    backup_count: int = 10,
) -> None:
    """Log every process event to the console and a rotating JSONL file."""

    resolved_dir = log_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        resolved_dir / "moto_laps.jsonl",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[console, file_handler], force=True)
    # HTTP requests are logged by our middleware with duration and correlation ID.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    logging.raiseExceptions = False
