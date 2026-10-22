"""Live-database tests (run in CI against pgvector/pgvector:pg16). ``pytest -m db``."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DBAPIError

pytestmark = pytest.mark.db

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    eng = create_engine(url)
    yield eng
    eng.dispose()


def test_tables_and_extension_exist(engine: Engine) -> None:
    with engine.connect() as conn:
        tables = set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            ).scalars()
        )
        ext = conn.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).scalar()
    expected = {
        "projects",
        "documents",
        "chunks",
        "regulations",
        "requirements",
        "regulation_matches",
        "coverage_results",
        "review_items",
        "audit_events",
        "users",
        "jobs",
    }
    assert expected <= tables
    assert ext == "vector"


def test_audit_events_are_append_only(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO audit_events (event_type, actor, payload) VALUES ('test', 'pytest', '{}')"
            )
        )
    for stmt in (
        "UPDATE audit_events SET actor='x'",
        "DELETE FROM audit_events",
        "TRUNCATE audit_events",
    ):
        with pytest.raises(DBAPIError, match="append-only"), engine.begin() as conn:
            conn.execute(text(stmt))
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM audit_events")).scalar() == 1


def test_tsvector_is_generated_and_vectors_of_any_dim_store(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO regulations (id, source, citation, part, title, text, embedding, embedding_model)"
                " VALUES (gen_random_uuid(), 'FAR', '52.204-21', '52', 'Basic Safeguarding',"
                " 'The Contractor shall apply basic safeguarding requirements', '[0.1,0.2,0.3]', 'mock/3')"
            )
        )
        hit = conn.execute(
            text(
                "SELECT citation FROM regulations WHERE tsv @@ plainto_tsquery('english', 'safeguarding')"
            )
        ).scalar()
    assert hit == "52.204-21"
