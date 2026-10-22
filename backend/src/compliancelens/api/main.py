"""FastAPI application entry point. Feature routers are added in later phases."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sqlalchemy import text

from compliancelens import __version__
from compliancelens.config import get_settings
from compliancelens.db.session import get_engine
from compliancelens.logging_setup import configure_logging
from compliancelens.tracing import setup_tracing


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.secret_values())
    setup_tracing(settings)
    yield


app = FastAPI(title="ComplianceLens", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus a DB probe and the active (non-secret) provider config."""
    settings = get_settings()
    db_ok = True
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "version": __version__,
        "database": db_ok,
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }
