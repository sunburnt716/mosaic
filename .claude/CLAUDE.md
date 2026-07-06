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

**Currently building:** the ingestion engine is complete — adapters (RSS, REST-JSON),
normalizer, dedup, quality gate, storage (raw store, seen store, poll state), and the
concrete `Engine` wired into the scheduler (`run.py`). The **processing extraction
engine** (downstream of storage) is also underway: Phase 0 (document-type inference +
validation) and Phase 1 (chunking — Documents → `Chunk`s by inferred `document_type`)
are in place; see "Processing layer" below. Ingestion and processing were built as two
parallel workstreams and merged — see "Document schema: two type fields" below for the
one contract point where that merge required a deliberate reconciliation.

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

## Ingestion engine
Complete: `adapters/` (RSS, REST-JSON — format-driven, one implementation per format,
never per source) → `pipeline/` (`normalizer`, `dedup`, `hashing`, `quality`, `transforms`,
`validation`) → `storage/` (`raw_store`, `seen_store`, `poll_state`), orchestrated by
`ConcreteEngine` (`ingestion/engine.py`), which satisfies the `Engine` Protocol that
`run.py`'s scheduler (`tick()`/`run_forever()`) depends on.

- **`ConcreteEngine.process_source(source)` handles exactly one source** and returns
  `None`. It does not loop over sources and does not check `enabled` — multi-source
  dispatch, disabled-source skipping, and per-source failure isolation across a tick are
  `run.py`'s job (`select_due_sources`, `tick`), not the engine's. `process_source` only
  catches the fetch-boundary signals it knows how to handle (`NotModifiedSignal`,
  `TransportError`, `FetchError`) and per-record `NormalizationError`s (dropped and
  counted); anything else propagates to `tick()`'s own isolation.
- **Adapters emit a fixed standard shape** (`url`, `title`, `raw_body`, `published`,
  `source_article_id`, `raw_payload`) regardless of source. `SourceConfig.field_mappings`
  is a per-source *override* for the rare source whose entries don't fit — normally `{}`.
- **SEC EDGAR: no specialized adapter.** A dedicated adapter querying EDGAR's full-text
  search (`efts.sec.gov`) was tried and retired — that endpoint has field-name mismatches
  and returns no filing bodies (see `adapters/edgar.py`, kept as a documented placeholder).
  EDGAR discovery instead runs through the generic RSS adapter pointed at the `getcurrent`
  Atom feed, with the `edgar_filing_url` transform (`pipeline/transforms.py`) cleaning the
  entry title. `"edgar"` is deliberately **not** a valid `SourceConfig.adapter` value.
- **Quality gate (`pipeline/quality.py`) is advisory only** — it warns on collapsed
  batches, empty-body rates, fallback titles, etc., but never drops or blocks. Per-source
  thresholds (`SourceConfig.max_fallback_title_rate`, `max_empty_body_rate`, `min_records`,
  `expects`) tune it; unset, the gate's source-agnostic defaults apply.
- **Transport validation (`pipeline/validation.py`) is fail-closed** — a structurally
  broken batch (HTML challenge page, malformed feed) is refused whole via `TransportError`
  before it ever reaches normalize/dedup/store. This is a different layer from the quality
  gate: transport judges the *batch's format*, quality judges *content patterns*.
- **Runtime deps are real now**: `requirements.txt` has `pyyaml`, `feedparser`, `requests`,
  `transformers` (chunk-sizing tokenizer). `requirements-dev.txt` has `pytest`,
  `pytest-mock`, `ruff`. Adapters import `feedparser`/`requests` lazily inside their fetch
  methods, so the unit suite runs without them installed — only live-network or
  integration runs need `requirements.txt`.

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

### Document schema: two type fields (ingestion + processing reconciliation)
Ingestion and processing were built as parallel workstreams with their own Document
type field, then merged. Rather than pick one, both fields exist because they answer
different questions:
- **`Document.doc_type`** — stamped verbatim at ingest from `SourceConfig.doc_type`
  (an ingest-time field, alongside `tickers`/`sectors`/`key_points`/`status`). It is
  the human-authored *advisory hint* (the "source signature map"): a source config
  declares "this feed is filings" or "this feed is articles". Vocabulary is only
  `article`/`filing` — it cannot express "tweet".
- **`Document.document_type`** (optional, `None` until Phase 0 inference runs) is the
  **authoritative** per-document type that Phase 1 chunking dispatches on. It is
  content-inferred, richer (`article`/`filing`/`tweet`/`unknown`), and may **override**
  `doc_type` when structure strongly disagrees with the advisory.
- **Never conflate them.** `doc_type` is what the source *claims*; `document_type` is
  what Phase 0 *concluded*. The chunking registry keys on `document_type`, never
  `doc_type` — a source misconfigured as `article` that's actually posting tweets still
  chunks correctly once inference overrides it.

### Phase 0 decisions (document-type inference + validation)
- **Heuristic only — no model.** Type detection uses algorithmic signals (token count,
  distinct filing markers, paragraph structure). LLMs/embeddings are Phase 2+ and are
  for embeddings, never for typing. This is a hard constraint.
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

### Phase 1 decisions (chunking)
- **Dispatch by inferred `document_type`.** `engine.chunk_document(doc)` → `chunkers/
  registry.get_chunker(doc.document_type)`: `article → chunk_paragraph`, `filing →
  chunk_section`, everything else (`tweet`, `unknown`, or `None` before inference) →
  `chunk_fixed` (fallback). The registry maps the `type_inference` constants, so the
  chunking vocabulary can't drift from inference's.
- **Dual spans are load-bearing.** Every `Chunk` carries `full_span` (whole chunk, for
  embedding/retrieval) and `highlight_span` (first-sentence excerpt, for citation) plus
  provenance (`title`/`url`/`source_name`/`tier`/`published_date`) copied off the parent
  Document so it survives into Chroma. Construction lives in `chunk.build_chunk` /
  `materialize_chunks` — copied once, never per chunker.
- **Reuse, don't re-derive structure.** The paragraph chunker splits on
  `text_metrics.paragraph_spans` (offset sibling of `count_paragraphs`); section
  detection reuses `text_metrics.FILING_MARKER_PATTERNS`. Structure is defined once so
  inference, validation, and chunking never disagree.
- **MiniLM tokenizer for chunk *sizing* (deliberate exception).** Phase 1 sizes/slices
  chunks with the real MiniLM tokenizer (`processing/utils/tokenization.py`, lazy-loaded +
  cached; adds `transformers`), so window sizes align with the Phase 2 embedder. This is a
  *distinct* notion of "token" from `text_metrics.count_tokens` (the Phase-0 whitespace
  proxy) and is the one place a model appears before Phase 2 — chunk sizing only, never
  type detection (that stays heuristic). Chosen over the whitespace proxy on purpose.
- **Pure, no I/O.** Chunking takes in-memory Documents and returns Chunks. The offline
  test suite injects a word-level fake tokenizer (`tests/processing/conftest.py`) so it
  never downloads MiniLM or imports `transformers`.

## Known gaps
- **`feedparser` may be unavailable in sandboxed dev environments** (its `sgmllib3k`
  dependency can fail to build without full PyPI network access). Adapters import it
  lazily inside `RssAdapter._fetch_feed`, so this only affects the handful of tests that
  exercise the real feed parse (`test_adapter_contract.py`'s conditional-GET/transport
  classes, `test_quality_fixtures.py`, `test_transforms.py`'s fixture-roundtrip class,
  all of `test_validation.py`) — everything else in the suite is unaffected. Install
  `requirements.txt` in an environment with full network access to run those.
- **`tests/test_sources.py` and `tests/test_integration.py` were not ported** from the
  parallel ingestion workstream: both depend on a JSON-based source registry
  (`ingestion/sources.py` + a repo-root `config/sources.json`) that was superseded by
  this branch's YAML registry (`ingestion/config/sources.yaml` + `core.source_config.
  load_sources`) — two competing registries would conflict. `test_source_config.py`
  already covers the YAML registry's loader/validation.

## Testing

Tests are the executable spec — write them before or alongside implementation,
never after. Every pipeline stage must have tests before the code ships.

### Structure
```
tests/
  conftest.py              # shared builders (make_document, make_source_config) + FakeResponse
  fixtures/                # captured payloads (raw bytes) + parsed-dict samples per source
    rss_reuters_sample.json    rest_json_sample.json   rest_json_raw.json
    sec-edgar.xml              ft-rss.xml
    challenge_page.html        degenerate_feed.xml
  test_hashing.py          # content_hash, identity_key, document_id contracts
  test_normalizer.py       # normalize() contracts: field mapping, HTML strip, date/URL validation
  test_dedup.py            # classify() contracts: L1/L2/L3/NEW result correctness
  test_adapter_contract.py # Adapter contracts: dict shape, FetchError, conditional GET, transport
  test_validation.py       # transport + parse fail-closed checks
  test_quality.py          # quality gate: flags, configurable thresholds, QualityReport/stats
  test_quality_fixtures.py # gate regression: degenerate feed warns, healthy EDGAR silent
  test_transforms.py       # per-source transforms (edgar_filing_url exact URL)
  test_engine.py           # end-to-end: dedup branches, 304, transport rejection, drop-and-count
  processing/              # processing layer tests (type inference, validation, chunking)
```

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

## Known follow-ups (flagged, not yet addressed)
- `rag_architecture.mermaid` is referenced under Architecture but not yet committed.
- The `schema-guardian` agent's documented field list has drifted from `ingestion/core/document.py`
  (it says `article_id`/`fetched_date`; the code uses `id`/`fetched_at`). Code is source of truth;
  reconcile the agent doc before relying on it for reviews.

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
