"""Database engine, session factory, and declarative base.

Provides a FastAPI dependency (:func:`get_db`) and a small context-manager
helper (:func:`session_scope`) for use in scripts and background tasks.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings


class Base(DeclarativeBase):
    """Declarative base class shared by all ORM models."""


_IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")

# --- SQLite (development default) -----------------------------------------
# ``check_same_thread`` is required for SQLite when accessed across threads
# (FastAPI runs sync endpoints in a threadpool; background tasks add more).
_engine_kwargs = {"future": True}
if _IS_SQLITE:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # --- PostgreSQL (production) ------------------------------------------
    # Connection pooling for concurrent traffic. Requires a driver, e.g.
    #   pip install "psycopg[binary]"      # sync driver
    #   DATABASE_URL=postgresql+psycopg://user:pass@host:5432/macromancer
    # For a fully async stack, use asyncpg with create_async_engine instead:
    #   pip install asyncpg
    #   DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/macromancer
    #   from sqlalchemy.ext.asyncio import create_async_engine
    #   engine = create_async_engine(URL, pool_size=20, max_overflow=10,
    #                                pool_pre_ping=True, pool_recycle=1800)
    _engine_kwargs.update(
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
    )

engine = create_engine(settings.DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables. Safe to call repeatedly (idempotent)."""
    # Import models so they register with ``Base.metadata`` before create_all.
    from backend import models  # noqa: F401  (side-effect import)

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a database session and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager providing a transactional session for scripts/tasks."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
