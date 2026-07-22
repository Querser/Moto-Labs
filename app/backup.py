"""Consistent rotating backups for the local SQLite race database."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import make_url

logger = logging.getLogger(__name__)


def backup_sqlite_database(
    database_url: str,
    backup_dir: Path,
    *,
    keep: int = 10,
) -> Path | None:
    """Create a transactionally consistent backup and rotate old copies."""

    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return None
    source_path = Path(url.database).expanduser().resolve()
    if not source_path.is_file():
        return None

    destination_dir = backup_dir.expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = destination_dir / f"moto_laps_{stamp}.db"
    temporary = destination.with_suffix(".tmp")
    try:
        # sqlite3.Connection's context manager commits but does not close the
        # Windows file handle, so explicit closing is required before os.replace.
        with (
            closing(sqlite3.connect(source_path)) as source,
            closing(sqlite3.connect(temporary)) as target,
        ):
            source.backup(target)
        os.replace(temporary, destination)
        backups = sorted(destination_dir.glob("moto_laps_*.db"), reverse=True)
        for obsolete in backups[max(1, keep) :]:
            obsolete.unlink(missing_ok=True)
        logger.info("Database backup created", extra={"backup_path": str(destination)})
        return destination
    except Exception:
        logger.exception("Database backup failed", extra={"database_path": str(source_path)})
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.warning("Temporary backup file is still locked", extra={"path": str(temporary)})
        return None
