"""Database engine and session management.

SQLite needs two things configured explicitly or it will quietly misbehave
under a scheduler plus a dashboard plus a Telegram bot all touching the same
file: WAL journaling (so a reader never blocks the writer) and foreign-key
enforcement (off by default in SQLite).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from investment_box.core.logging import get_logger
from investment_box.db.models import Base

log = get_logger(__name__)


def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
    """Per-connection PRAGMAs applied to every SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    # Wait rather than fail immediately when another process holds the write lock.
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


class Database:
    """Owns the engine and hands out sessions."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        is_sqlite = url.startswith("sqlite")

        if is_sqlite and ":memory:" not in url:
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

        self.engine: Engine = create_engine(
            url,
            echo=echo,
            future=True,
            # check_same_thread=False: APScheduler runs jobs on worker threads.
            connect_args={"check_same_thread": False} if is_sqlite else {},
        )
        if is_sqlite:
            event.listen(self.engine, "connect", _configure_sqlite)

        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def create_all(self) -> None:
        """Create any missing tables.

        Fine for a fresh install and for tests. Schema *changes* go through
        alembic so that an existing database with live trade history is never
        silently reshaped.
        """
        Base.metadata.create_all(self.engine)
        log.debug("schema.ensured", url=self.url)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A transactional session: commits on success, rolls back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()


_database: Database | None = None


def get_database(url: str | None = None, *, echo: bool = False) -> Database:
    """Process-wide database handle.

    Pass ``url`` explicitly in tests; the singleton is for application entry
    points, which must all share one engine.
    """
    global _database
    if url is not None:
        return Database(url, echo=echo)
    if _database is None:
        from investment_box.config import get_settings

        _database = Database(get_settings().db_url, echo=echo)
        _database.create_all()
    return _database


def reset_database_singleton() -> None:
    """Drop the cached handle. Tests only."""
    global _database
    if _database is not None:
        _database.dispose()
    _database = None
