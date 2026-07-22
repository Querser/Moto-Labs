"""Database engine and transactional session helpers."""

from __future__ import annotations

import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, event
from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.models import Base

DEFAULT_DATABASE_URL = "sqlite:///data/moto_laps.db"
# RACE_DATABASE_URL remains a compatibility override for older installations.
DATABASE_URL = os.getenv("RACE_DATABASE_URL", get_settings().database_url)


def _ensure_sqlite_parent(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        return
    if url.database == ":memory:":
        return
    Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _configure_sqlite(dbapi_connection: object, _connection_record: object) -> None:
    """Enable integrity and practical concurrency settings on every SQLite connection."""

    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        # WAL is not available for transient in-memory databases and SQLite safely
        # ignores the requested mode there.
        cursor.execute("PRAGMA journal_mode=WAL")
        # Use SQLite's own page cache instead of an application cache that could
        # return stale lap data after OCR writes or manual corrections.
        cursor.execute("PRAGMA cache_size=-20000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
    finally:
        cursor.close()


def create_db_engine(database_url: str | None = None, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy 2.x engine with safe SQLite defaults."""

    url = database_url or DATABASE_URL
    _ensure_sqlite_parent(url)
    parsed = make_url(url)
    kwargs: dict[str, object] = {"echo": echo, "future": True}
    if parsed.get_backend_name() == "sqlite":
        kwargs["connect_args"] = {"check_same_thread": False}
        if parsed.database in (None, "", ":memory:"):
            kwargs["poolclass"] = StaticPool
    engine = sqlalchemy_create_engine(url, **kwargs)
    if parsed.get_backend_name() == "sqlite":
        event.listen(engine, "connect", _configure_sqlite)
    return engine


engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def make_session_factory(bind: Engine) -> sessionmaker[Session]:
    """Return a test/integration-friendly session factory for a supplied engine."""

    return sessionmaker(bind=bind, autoflush=False, expire_on_commit=False)


def init_db(bind: Engine | None = None) -> None:
    """Create all tables for first-run convenience.

    Production upgrades should use Alembic.  Keeping this function idempotent makes
    the local application boot reliably on a brand-new machine.
    """

    Base.metadata.create_all(bind or engine)


def upgrade_database(database_url: str | None = None) -> None:
    """Apply committed Alembic migrations for normal application startup."""

    config_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    config = Config(str(config_path))
    # The running application already owns structured console/file handlers;
    # Alembic must not replace them with its CLI-only logging configuration.
    config.attributes["configure_logger"] = False
    config.set_main_option("sqlalchemy.url", (database_url or DATABASE_URL).replace("%", "%%"))
    command.upgrade(config, "head")


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """Provide an atomic unit of work which commits or rolls back as a whole."""

    session = (factory or SessionLocal)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency which never leaks an open transaction or connection."""

    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Compatibility name for callers that prefer domain wording.
get_session = get_db


__all__ = [
    "DATABASE_URL",
    "DEFAULT_DATABASE_URL",
    "SessionLocal",
    "create_db_engine",
    "engine",
    "get_db",
    "get_session",
    "init_db",
    "make_session_factory",
    "session_scope",
    "upgrade_database",
]
