"""Verify provider wiring with one tiny structured call (and a cache hit on re-run).

Usage (from backend/):
    uv run python scripts/llm_smoke.py                 # uses LLM_PROVIDER from .env
    LLM_PROVIDER=mock uv run python scripts/llm_smoke.py
    uv run python scripts/llm_smoke.py --embeddings    # also embed a passage + query
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Literal

from pydantic import BaseModel, Field

from compliancelens.config import LLMTask, get_settings
from compliancelens.embeddings import build_embedder
from compliancelens.llm import build_llm
from compliancelens.llm.audit import InMemoryAuditSink
from compliancelens.logging_setup import configure_logging


class ModalityGuess(BaseModel):
    modality: Literal["shall", "must", "will", "should", "may"]
    reason: str
    confidence: float = Field(ge=0, le=1)


SENTENCE = (
    "The Contractor shall deliver monthly status reports to the Contracting Officer's "
    "Representative no later than the fifth business day of each month."
)


async def main(check_embeddings: bool) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.secret_values())
    audit = InMemoryAuditSink()
    llm = build_llm(settings, audit=audit)
    prompt = llm.prompts.render("smoke_test", sentence=SENTENCE)
    for attempt in (1, 2):
        result = await llm.generate(task=LLMTask.FAST, prompt=prompt, output=ModalityGuess)
        print(f"--- call {attempt} ---")
        print("value:", result.value.model_dump() if result.value else None)
        print("record:", json.dumps(result.record.model_dump(), indent=2))

    if check_embeddings:
        embedder = build_embedder(settings)
        passage = (await embedder.embed_passages([SENTENCE]))[0]
        query = await embedder.embed_query("monthly status report deadline")
        cos = sum(a * b for a, b in zip(passage, query, strict=True))
        print(f"embeddings: model={embedder.model_name} dim={len(passage)} cosine={cos:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true")
    asyncio.run(main(parser.parse_args().embeddings))
