# ComplianceLens — Build Plan

Federal RFP & grant compliance copilot: extract every requirement, ground it in the
source PDF, map it to FAR / 2 CFR 200, check a draft proposal against it, and route
anything uncertain to a human. Runs at $0 with `docker compose up`.

## Guiding principles

1. **Grounded or flagged.** Every model claim (quote, page, citation) is checked
   deterministically against source data. If the check fails, the item goes to a human.
2. **Auditable.** Every LLM call, retrieval and human decision is an append-only row.
3. **Provider-neutral.** `nvidia | anthropic | mock` all return the same validated
   Pydantic objects. CI never touches the network.
4. **Untrusted input.** Document text is data, never instructions.

## Repository layout

```
.
├── docker-compose.yml          # postgres+pgvector, phoenix, api (and web in phase 6)
├── .env.example                # blank secrets, documented knobs
├── .github/workflows/ci.yml    # ruff, mypy, pytest (mock), alembic vs pgvector service
├── backend/
│   ├── pyproject.toml / uv.lock
│   ├── Dockerfile
│   ├── alembic.ini, alembic/   # migrations (hand-written, reviewed)
│   ├── prompts/                # versioned prompt templates  <name>.v<N>.prompt
│   ├── scripts/                # fetch_regulations, generate_sample_rfps, evals, llm_smoke
│   ├── src/compliancelens/
│   │   ├── config.py           # pydantic-settings; secrets as SecretStr
│   │   ├── logging_setup.py    # secret-redacting log filter
│   │   ├── tracing.py          # OpenTelemetry -> Phoenix
│   │   ├── db/                 # SQLAlchemy 2 models + session
│   │   ├── llm/                # provider abstraction, rate limiter, backoff, cache, prompts
│   │   ├── embeddings/         # nvidia | local | mock
│   │   ├── ingest/             # (P2) PyMuPDF parsing, chunking
│   │   ├── retrieval/          # (P2) hybrid search + RRF + reranker
│   │   ├── extraction/         # (P3) requirements, grounding, dedupe
│   │   ├── mapping/            # (P4) regulation mapping
│   │   ├── coverage/           # (P4) proposal coverage
│   │   ├── review/             # (P5) queue, routing rules, agreement metrics
│   │   ├── audit/              # (P5) audit sink + export
│   │   ├── export/             # (P5) xlsx + docx
│   │   └── api/                # FastAPI routers
│   └── tests/
└── frontend/                   # (P6) React + Vite + TS + Tailwind + react-pdf
```

## Phases (stop after each for test + commit)

### Phase 1 — Skeleton & provider layer
- docker-compose: `pgvector/pgvector:pg16`, Arize Phoenix, API container.
- Config via pydantic-settings; `.env.example`; secret-redacting logging.
- LLM layer: `StructuredLLM` facade → `ChatProvider` (nvidia / anthropic / mock).
  - NVIDIA: `openai` SDK, `base_url=https://integrate.api.nvidia.com/v1`, structured
    output via `nvext.guided_json` (switchable to `json_schema` / tool calling).
  - Anthropic: official SDK, `output_config.format` JSON schema, prompt caching on
    system + regulation context blocks.
  - Mock: deterministic per-prompt handlers + schema stubs + optional fixture files.
  - Sliding-window RPM limiter, full-jitter exponential backoff on 429/5xx
    (honours `Retry-After`), disk cache keyed on hash(provider, model, prompt, schema).
  - Pydantic validation with one repair retry, then `needs_review`.
  - Every call yields an `LLMCallRecord` (tokens, latency, cost, cache hit) for the audit sink.
- Embeddings: nvidia `nv-embedqa-e5-v5` (passage/query input types), local bge-small,
  mock hashing embedder. Model name stored with every vector.
- Prompt registry: `prompts/<name>.v<N>.prompt` with `system / context / user` sections,
  Jinja2 StrictUndefined, `untrusted` filter that wraps document text in nonce'd delimiters.
- Full data model + initial Alembic migration (incl. append-only trigger on `audit_events`).
- FastAPI app with `/health`. CI pipeline.

### Phase 2 — Data & retrieval
- `scripts/fetch_regulations.py`: eCFR versioner API → XML → clause records
  (FAR Parts 9, 15, 19, 52; 2 CFR 200) → JSON cache → DB load + embed.
- `scripts/generate_sample_rfps.py`: 3 synthetic PDFs (IT modernization, cyber services,
  research NOFO) with planted requirements + `ground_truth.json`; one variant with an
  embedded prompt-injection line.
- Ingestion: PyMuPDF blocks w/ bbox, heading hierarchy from font size/weight + numbering,
  structure-aware chunking (400–800 tokens, overlap), embeddings + tsvector.
- Hybrid retrieval (pgvector cosine + `ts_rank_cd`) fused by RRF; optional NVIDIA reranker.
- Per-model partial HNSW indexes so vectors from different models never mix.
- `scripts/eval_retrieval.py`: recall@k, MRR.

### Phase 3 — Requirement extraction
- Per-section structured extraction → `Requirement` list.
- Grounding: rapidfuzz `partial_ratio` ≥ 90 of `verbatim_quote` vs text on the cited page
  (also searches ±1 page to report a corrected page, but still flags).
- Dedupe: normalized text + embedding similarity, keep the earliest, link `duplicate_of`.
- Prompt-injection test PDF; test that output is unchanged.

### Phase 4 — Mapping & coverage
- Top-k regulation retrieval per requirement → `RegulationMatch`; citations validated
  by lookup, invented ones dropped and the item flagged.
- Proposal coverage: retrieve proposal passages, classify addressed / partial / not,
  evidence quote grounded on page, suggested fix, risk score → sorted gap list.

### Phase 5 — Review, audit, exports (backend)
- Routing rules (confidence, grounding, invalid citation, security/pricing always).
- Decision endpoint (approve / edit / reject + reason), before/after snapshots.
- Agreement rate per category.
- DB audit sink, `/audit/{project_id}` JSON export, trigger-enforced append-only.
- `export.xlsx` (colour-coded, page hyperlinks) and Word summary memo.
- Jobs table + BackgroundTasks for extract / coverage.

### Phase 6 — React UI
- Project list, upload, job progress.
- Requirements table with filters (status, category, modality, flags).
- Review screen: requirement + rationale + citations | react-pdf with bbox highlight.
- Playwright smoke test against mock provider.

### Phase 7 — Evals & README
- Extraction P/R/F1 vs ground truth per provider; retrieval recall@k / MRR; cost/doc.
- README: Mermaid architecture, anti-hallucination guard, Security & Safety
  (prompt injection, data residency, why human review), GovCloud / Azure Government mapping.

## Licence notes (all free to use)
- PyMuPDF is **AGPL-3.0** — fine for a portfolio/internal tool; a commercial closed
  deployment would need Artifex's commercial licence or a swap to `pypdfium2`.
- Arize Phoenix server is **Elastic License 2.0** (free, source-available, not OSI). We only
  run its Docker image; Python side uses Apache-2.0 OpenTelemetry/OpenInference packages.
- Everything else: MIT / BSD / Apache-2.0 / PostgreSQL licence.
