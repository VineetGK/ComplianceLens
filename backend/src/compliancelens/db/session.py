"""Engine and session factory."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from compliancelens.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide engine for ``DATABASE_URL``."""
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a session that is closed afterwards."""
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    with factory() as session:
        yield session
