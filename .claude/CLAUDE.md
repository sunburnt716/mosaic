# CLAUDE.md

## Project
Investing-news RAG system. Mission: **inform, not advise** — surface
market-moving news with cited, timestamped sources so investors stay aware
without doing their own research. Never produces buy/sell/hold calls.

## Architecture
Full diagram: `rag_architecture.mermaid` — read it first. Pipeline, in order:

sources (tiered) → format adapters → normalizer → dedup → raw store →
processing (chunk + embed) → Chroma → retrieval (search + re-rank + cluster)
→ generative models → interfaces.

**Currently building:** the ingestion engine (everything up to and including dedup)
and the **processing extraction engine** (downstream of storage). Processing Phase 0
(document-type inference + validation) is in place; see "Processing layer" below.

## Core principles (always)
- **Structure mirrors the pipeline.** File/function layout follows data flow,
  not arbitrary groupings.
- **Contracts come first.** Define schemas/interfaces before implementation.
  The normalized document schema is *the* contract — don't break it casually.
- **Config over code.** Per-source behavior lives in config, not per-source
  code paths. Adding a source = adding config, not code.

## Stack
- Python 3.11+ (pinned `requires-python = ">=3.11"` in `pyproject.toml`)
- pip + venv for environments; dependencies pinned in `requirements.txt`
- Chroma (vector store); MiniLM (local) or Gemini for embeddings
- Groq / Llama 3.1 8B (light: classify/route/borderline dedup);
  Gemini Flash (synthesis)
- ruff (lint + format); pytest (tests)

## Commands
```bash
python -m venv .venv && source .venv/bin/activate   # one-time setup
pip install -r requirements.txt                      # install dependencies
ruff check .                                         # lint
ruff format .                                        # format
pytest                                               # run tests
# run: <fill in once an entrypoint exists>
```

## Conventions
- One embedding model per Chroma collection — never mix models in a collection.
- Tier (trust level) is stamped at ingest, on the source — never inferred later.
- Dedup is three distinct levels; don't collapse them:
  L1 exact bytes (content hash), L2 same-article-updated (identity key),
  L3 same-story-cross-outlet (embeddings).
- Keep the raw payload untouched in the store so downstream stages can re-run offline.
- Prefer adjusting config over writing new per-source code.
- Polling is config-driven and stateful: `SourceConfig.poll_interval` (static, in
  `sources.yaml`) says how often; `PollStateStore` (runtime `poll_state.json`) records
  last-polled time + ETag/Last-Modified for conditional GETs. `run.py` is the single
  scheduler that reads due-ness; the same scheduler later drives hot-path processing.

## Processing layer (Extraction Engine)
The processing layer sits **downstream of storage**: it reads normalized Documents,
infers their type, chunks by type, embeds, and writes vectors to Chroma. It is a
**separate top-level package** (`processing/`, sibling of `ingestion/`) that ingestion
and query-time retrieval both call into — never logic baked into ingestion.

**Folder rule:** ALL processing logic lives under `processing/`, even the parts that
run *during* the ingestion window (the hot path). Processing logic is never split
between `ingestion/` and `processing/`. This keeps the structure clean and avoids
merge conflicts between the two parallel workstreams.

**Two trigger paths** (wired in later phases, not hardcoded by tier):
- **Hot path** — server-side, inside the existing single scheduler (`run.py`):
  high-frequency sources are chunked + embedded right after normalization.
- **Cold path** — on-demand at query time: a document not yet in Chroma is processed
  in real time.
The hot/cold split is driven by **processing throughput + per-source cost vs the
ingestion window**, not by tier number (Tier 0/1 usually hot, Tier 3 usually cold,
Tier 2 depends on velocity). One scheduler only — no second scheduler.

**Phase map:** 0 type inference + validation · 1 chunking (paragraph/section/fixed) ·
2 embeddings (MiniLM/Gemini — embeddings only, never type detection) · 3 Chroma write ·
4 wire into the scheduler (hot-path frequency/cost decision) · 5 query-time fallback.

### Phase 0 decisions (document-type inference + validation)
- **Heuristic only — no model.** Type detection uses algorithmic signals (token count,
  distinct filing markers, paragraph structure). LLMs/embeddings are Phase 2+ and are
  for embeddings, never for typing. This is a hard constraint.
- **`Document.document_type`** (new, optional, `None` until inferred) is the
  authoritative per-document type that Phase 1 chunks by. It is **distinct from
  `SourceConfig.doc_type`**, which is only the human-authored *advisory hint*
  (the "source signature map") — inference may consult and **override** it. Two
  different names for two different things (inferred result vs config declaration);
  there is exactly one type field on the Document — no redundancy.
- **Advisory vocabulary gap:** `SourceConfig.doc_type` is only `article`/`filing`, so
  genuine tweets are caught by structure, not config. Reconciliation policy: a *strong*
  structural signal (filing markers + length; or confidently tiny text) overrides the
  advisory; otherwise the advisory wins the ambiguous middle; otherwise fall back to
  the structural guess or `unknown`.
- **`Document.validation_warnings`** (new, optional) carries structure-vs-type warnings.
  Both new fields are the **documented exception** to the "derive at ingest, never infer
  from content" rule: they are populated by processing, are purely informational, and
  **never gate citation or dedup** (those depend only on ingest-time fields).
- **Validation never raises and never blocks.** It returns
  `ValidationResult(is_valid, warnings, severity)` where `severity` is
  `info < warning < degenerate` and `is_valid` is False only for `degenerate`. Even
  degenerate docs flow through; warnings are recorded for per-source quality monitoring.
  `unknown` docs are *deferred* (a warning, still valid).
- **Shared `processing/text_metrics.py`** owns token/paragraph/marker counting so
  inference and validation can never drift on those definitions (no duplicated logic).

## Guardrails (prefer X over Y)
- Prefer surfacing news + sources over generating advice. No buy/sell/hold, ever.
- Prefer citing source + timestamp on every claim over unattributed synthesis.
- Prefer Tier 0/1 sources for corroboration; treat Tier 3 as signal only.
- Prefer plain code + embeddings for routing/dedup before reaching for an LLM.

## RAG fitness (check before code lands)
- Chunk by doc type: filings by section, articles by paragraph. Wrong-grain
  chunks quietly wreck retrieval.
- Carry citation metadata all the way through: `source_name`, `url`, `tier`,
  `published_date` must survive into Chroma so synthesis can cite + timestamp.
- Re-ranking is your code, not Chroma's: blend semantic score with recency,
  source credibility, and user profile.
- Preserve cross-outlet corroboration (L3): retrieval should surface the same
  story from multiple sources so trust can be assessed, not collapse it.
- One embedding model per collection — a change that mixes models in a
  collection is a RAG-fitness failure, not just a style nit.
