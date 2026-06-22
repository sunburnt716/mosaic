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

**Currently building:** the ingestion engine (everything up to and including dedup).

## Core principles (always)
- **Structure mirrors the pipeline.** File/function layout follows data flow,
  not arbitrary groupings.
- **Contracts come first.** Define schemas/interfaces before implementation.
  The normalized document schema is *the* contract — don't break it casually.
- **Config over code.** Per-source behavior lives in config, not per-source
  code paths. Adding a source = adding config, not code.

## Stack
- Python <SET VERSION>
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
