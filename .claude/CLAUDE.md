# CLAUDE.md

## Project
Investing-news RAG system. Mission: **inform, not advise** — surface
market-moving news with cited, timestamped sources so investors stay aware
without doing their own research. Never produces buy/sell/hold calls.

## Architecture
Full diagram: `rag_architecture.mermaid` — read it first. *(Note: this file is referenced
aspirationally but is not yet in the repo.)* Pipeline, in order:

sources (tiered) → format adapters → normalizer → dedup → raw store →
processing (chunk + embed) → Chroma → retrieval (search + re-rank + cluster)
→ generative models → interfaces.

**Currently building:** the ingestion engine is complete (sources → adapters → validation →
normalizer → quality gate → dedup → raw store, handing off `status: unprocessed` Documents).
The processing stage (`extraction/`) is underway: **Phase 1 chunking is built** (Documents →
`Chunk`s, dispatched by `doc_type`); embedding into Chroma and enrichment are the next steps.

### Top-level layout (structure mirrors the pipeline stages)
```
config/sources.json   makeshift source registry (data, not code) — read via load_sources()
ingestion/            sources → adapters → normalizer → dedup → raw store   (active)
extraction/           processing: chunk (built) + embed + enrich            (chunking active)
generation/           retrieval + synthesis                                  (scaffold — non-goal)
source_validation/    authoring-time source onboarding/validation           (scaffold — non-goal)
tests/                offline, fixture-based pytest suite
```
`generation/` and `source_validation/` are placeholders (README + `__init__`) that mark future
stages. `extraction/` has its first stage built — **chunking** (`extraction/chunk.py`,
`chunkers/`, `utils/`, `engine.py`); embedding + enrichment remain scaffolded. Shared contracts
(`Document`, `SourceConfig`) live in `ingestion/core/` for now and will be promoted to a shared
package when a downstream stage actually needs them.

**Chunking (extraction Phase 1):** `engine.chunk_document(doc)` dispatches on `doc_type` via
`chunkers/registry.py` — `article → chunk_paragraph`, `filing → chunk_section`, any other type
→ `chunk_fixed` (fallback). Every `Chunk` carries dual spans (`full_span` for embedding/retrieval,
`highlight_span` for the cited excerpt) plus citation provenance copied from its parent Document.
Tokenization is centralized in `extraction/utils/tokenization.py` (lazy-loaded MiniLM, one source
of truth — chunkers never load their own tokenizer). Chunking is pure (no I/O); reading the raw
store and advancing `status` belongs with the later embedding stage.

**Handoff boundary:** ingestion's only output is normalized, deduped `Document` rows in the
raw store, stamped `status: "unprocessed"` (the `Document.status` default). Downstream stages
read from there on their own clock and advance the status; ingestion never calls them and never
imports them (guarded by `tests/test_handoff.py`). The store is the buffer between the two clocks.

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
python -m ingestion.run --once --log-level DEBUG   # run all enabled sources once
python -m ingestion.run --once --source sec-edgar    # run one source for debugging
python -m ingestion.run                              # scheduled loop (Ctrl-C to stop)
# Sources are declared in config/sources.json (default), loaded via ingestion.sources.load_sources().
# Override the registry path with --config PATH.
```

## Conventions
- One embedding model per Chroma collection — never mix models in a collection.
- Tier (trust level) is stamped at ingest, on the source — never inferred later.
- Dedup is three distinct levels; don't collapse them:
  L1 exact bytes (content hash), L2 same-article-updated (identity key),
  L3 same-story-cross-outlet (embeddings).
- Keep the raw payload untouched in the store so downstream stages can re-run offline.
- Prefer adjusting config over writing new per-source code.
- Sources are config entries in `config/sources.json`, never code. Adding one = a JSON entry
  (url, tier, adapter, optional `transform`/`expects`); the engine never onboards at runtime.
- Conditional GET is opportunistic: HTTP adapters raise `NotModifiedSignal` on 304 and attach
  `_etag`/`_last_modified` to items; the engine persists them to `poll_state` (24h validator TTL,
  one reader = run.py decides who's due, one writer = engine writes after each poll). Sources that
  send no validators (e.g. EDGAR getcurrent, CNBC) simply re-fetch — that's expected, not a bug.
- Validation is layered by failure mode (`ingestion/pipeline/validation.py`):
  **transport + parse are fail-closed** — a structurally-broken batch (empty body, HTML challenge
  page where a feed/JSON was expected, malformed XML) raises `TransportError` and the whole batch
  is refused (`rejected_transport: true`), never reaching the store. **The per-record contract is
  drop-and-count** — a single bad record (missing/unparseable URL, bad date) is dropped via
  `NormalizationError` and counted in `dropped_records`; the rest of the batch survives.
- The **quality gate** (`ingestion/pipeline/quality.py`) is the fourth, softest layer: it runs on
  the normalized batch **before dedup** and **only warns — never drops, classifies, or routes**.
  It computes source-agnostic red flags (fallback-title / empty-body rates, URL / identity-key /
  content-hash collapse, malformed URLs, empty batch) and returns a `QualityReport(warnings, stats)`;
  both land in the run summary (`quality_warnings`, `quality_stats`). Per-source tuning is via
  optional `SourceConfig` thresholds (`max_fallback_title_rate`, `max_empty_body_rate`,
  `min_records`), applied only where set — never branch on a source name inside the gate.

## Known follow-ups (flagged, not yet addressed)
- `rag_architecture.mermaid` is referenced under Architecture but not yet committed.
- The `schema-guardian` agent's documented field list has drifted from `ingestion/core/document.py`
  (it says `article_id`/`fetched_date`; the code uses `id`/`fetched_at`). Code is source of truth;
  reconcile the agent doc before relying on it for reviews.
- `requirements.txt` still contains unresolved merge-conflict markers (`<<<<<<< HEAD` … `>>>>>>> main`)
  committed in `137bffa` — `pip install -r requirements.txt` fails as-is. Resolve toward the `main`
  side (the full dependency set) and add `transformers` (needed by the chunking tokenizer; it also
  arrives transitively via `sentence-transformers`). Pre-existing; not touched by the chunking work.
- Section chunking has no real fixture yet: EDGAR discovery is metadata-only (`expects.body = false`),
  so its Documents have empty bodies and produce no section chunks. `test_chunk_section.py` uses a
  synthetic filing body; add a captured full-text filing fixture when a full-text source is onboarded.
- The Phase 1 spec's *example* registry maps `news_article → chunk_fixed`, but the pipeline's actual
  `doc_type` values are `article`/`filing`, and this project's rule is "articles by paragraph" — so
  `article → chunk_paragraph`. Fixed-size is the fallback for unmapped/unstructured types.

## Guardrails (prefer X over Y)
- Prefer surfacing news + sources over generating advice. No buy/sell/hold, ever.
- Prefer citing source + timestamp on every claim over unattributed synthesis.
- Prefer Tier 0/1 sources for corroboration; treat Tier 3 as signal only.
- Prefer plain code + embeddings for routing/dedup before reaching for an LLM.

## Testing

Tests are the executable spec — write them before or alongside implementation,
never after. Every pipeline stage must have tests before the code ships.

### Structure
```
tests/
  conftest.py              # shared builders (make_document, make_source_config, make_chunk)
                           #   + FakeResponse (network) + FakeTokenizer/fake_tokenizer (chunking)
  fixtures/                # captured payloads (raw bytes) + parsed-dict samples per source
    rss_reuters_sample.json    rest_json_sample.json   rest_json_raw.json
    sec-edgar.xml              ft-rss.xml
    challenge_page.html        degenerate_feed.xml
  test_hashing.py          # content_hash, identity_key, document_id contracts
  test_normalizer.py       # normalize() contracts: field mapping, HTML strip, date/URL validation
  test_dedup.py            # classify() contracts: L1/L2/L3/NEW result correctness
  test_adapter_contract.py # Adapter contracts: dict shape, FetchError, conditional GET, transport
  test_validation.py       # transport + parse fail-closed checks
  test_sources.py          # load_sources() loader + registry invariants
  test_quality.py          # quality gate: flags, configurable thresholds, QualityReport/stats
  test_quality_fixtures.py # gate regression: degenerate feed warns, healthy EDGAR silent
  test_transforms.py       # per-source transforms (edgar_filing_url exact URL)
  test_engine.py           # end-to-end: dedup branches, 304, transport rejection, drop-and-count
  # --- extraction / chunking (Phase 1) ---
  test_tokenization.py     # tokenize_document / token_spans / count_tokens contracts (fake tokenizer)
  test_section_detection.py# detect_section_headers heuristics: all-caps / numbered / keyword vs prose
  test_highlight.py        # select_highlight_span: first-sentence + header-skip + fallback
  test_chunk.py            # Chunk contract: chunk_id, provenance copy, materialize_chunks offsets
  test_chunk_fixed.py      # fixed windows: overlap, ordinals, highlight==full, ValueError guard
  test_chunk_paragraph.py  # paragraph merge (forward + tail fold-back), first-sentence highlight
  test_chunk_section.py    # section split, preamble, after-header highlight, oversized fallback
  test_chunk_registry.py   # get_chunker mapping: article→paragraph, filing→section, other→fixed
  test_chunk_engine.py     # chunk_document dispatch by doc_type; chunk_documents flatten
```

**Chunking tests are offline via `fake_tokenizer`** (a word-level `FakeTokenizer` injected as the
cached MiniLM tokenizer — mirrors `FakeResponse` for the adapters). Assert the *contract*
(spans locate text, ordinals contiguous, dual-span rules), not MiniLM's specific sub-word IDs.

**Fixture-regression convention:** a source's *fixture + expected `Document`s = its regression
test*. Onboarding a source produces its test as a byproduct (capture the raw payload, assert the
normalized output). The quality gate is pinned the same way — `test_quality_fixtures.py` runs a
degenerate and a healthy captured payload through the real adapter→normalize→gate chain. Every
test is offline; CI makes zero external requests.

### Rules (always)
- **Tests use fixtures, not live APIs.** Save real API responses as JSON in `tests/fixtures/`
  and test adapters against those. Live-API tests go in a separate `@pytest.mark.integration`
  class marked `@pytest.mark.skip(reason="Requires live network access")`.
- **One fixture per source.** `tests/fixtures/<source>_sample.json` holds what the adapter
  yields for a single article — not the raw HTTP response, but the parsed dict the adapter
  emits. Adding a source = add a fixture + add tests, no new code paths.
- **Test the contract, not the implementation.** Tests must pass regardless of internal
  refactors as long as the public function signature and behaviour don't change.
- **Builder functions, not inline construction.** Use `make_document()` and
  `make_source_config()` from conftest.py with keyword overrides for the field under test.
  This keeps tests readable and insulates them from schema field additions.
- **MockSeenStore, not the real SeenStore.** Use the `MockSeenStore` in `test_dedup.py`
  to test dedup logic in isolation. When SeenStore is implemented, add integration tests
  against it separately.
- **L3 tests require explicit embedding vectors.** Pass `embedding=[...]` to `classify()`
  in L3 tests. Use orthogonal vectors for "no similarity" and identical/near-identical
  vectors for "high similarity".

### What to test at each stage
| Stage | Test file | Key assertions |
|---|---|---|
| hashing.py | test_hashing.py | determinism, whitespace normalization, hex format, no dashes |
| normalizer.py | test_normalizer.py | HTML stripped, date UTC-aware, tier from config, raw_payload untouched, NormalizationError on bad input |
| dedup.py | test_dedup.py | L1 priority over L2, L3 never discards, NEW on empty store |
| adapters | test_adapter_contract.py | yields dicts, required fields present, FetchError (not bare Exception) |

### Commands
```bash
pytest                          # run all unit tests
pytest -m "not integration"     # skip live-API tests (default in CI)
pytest -m integration           # run live-API tests only (requires keys)
pytest tests/test_hashing.py    # run a single test file
pytest -v                       # verbose output
```

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
