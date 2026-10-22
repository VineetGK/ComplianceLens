# ComplianceLens

Federal RFP & grant compliance copilot: extracts every requirement from a solicitation,
grounds it in the source PDF, maps it to FAR / 2 CFR 200, checks a draft proposal against
it, and routes anything uncertain to a human reviewer. Runs locally at $0.

> Status: **Phase 1 of 7** (skeleton + provider layer). See [PLAN.md](PLAN.md).
> The full README (architecture, evals, security, GovCloud mapping) lands in phase 7.

## Quickstart

```bash
cp .env.example .env          # add NVIDIA_API_KEY from build.nvidia.com (free)
docker compose up --build     # postgres+pgvector :5432, phoenix :6006, api :8000
curl localhost:8000/health
```

Local development without Docker for the API:

```bash
cd backend
uv sync
uv run pytest                      # unit tests, mock providers, no network
uv run ruff check . && uv run mypy
LLM_PROVIDER=mock uv run python scripts/llm_smoke.py --embeddings
uv run python scripts/llm_smoke.py # real call with your .env provider; run twice to see the cache hit
```

DB tests need Postgres (`docker compose up db`), then `uv run pytest -m db`.

## Providers

| `LLM_PROVIDER` | Backend | Structured output | Notes |
|---|---|---|---|
| `nvidia` (default) | NVIDIA API Catalog via `openai` SDK | `nvext.guided_json` (or `json_schema` / `tool`) | free tier; client-side RPM limiter + backoff |
| `anthropic` | Claude via `anthropic` SDK | `output_config.format` JSON schema | prompt caching on system + regulation context |
| `mock` | deterministic, offline | schema stubs / handlers / fixtures | used by tests and CI |

All providers return the same validated Pydantic objects. Invalid output gets one repair
turn with the validation errors, then is marked `needs_review`. Every call (including disk
cache hits) produces an audit record with tokens, latency, estimated cost and cache status.

`EMBEDDING_PROVIDER=nvidia|local|mock` — `nvidia/nv-embedqa-e5-v5` (passage/query input
types), `BAAI/bge-small-en-v1.5` offline (`uv sync --extra local-embeddings`), or a hashing
embedder for CI. The embedding model name is stored with every vector.

## Windows note

On a very long checkout path, Windows may fail to load compiled wheels ("filename or
extension is too long"). Put the venv on a short path:
`set UV_PROJECT_ENVIRONMENT=C:\venvs\compliancelens`. If Smart App Control blocks mypy's
native parser, run `uv run mypy --no-native-parser`.
