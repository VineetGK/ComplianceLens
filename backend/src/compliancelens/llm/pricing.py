"""Estimated per-call cost. Prices in USD per million tokens.

Anthropic list prices as of 2026-10 (first-party API); verify before quoting.
NVIDIA API Catalog free tier is $0. Unknown models are costed at $0 and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

from compliancelens.llm.types import RawCompletion


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
}

FREE_PREFIXES = ("meta/", "nvidia/", "mistralai/", "mock")


def estimate_cost(completion: RawCompletion) -> float:
    """USD cost of ``completion`` from its token counts."""
    if completion.model.startswith(FREE_PREFIXES):
        return 0.0
    price = PRICES.get(completion.model)
    if price is None:
        return 0.0
    return (
        completion.input_tokens * price.input
        + completion.output_tokens * price.output
        + completion.cache_read_tokens * price.cache_read
        + completion.cache_write_tokens * price.cache_write
    ) / 1_000_000
