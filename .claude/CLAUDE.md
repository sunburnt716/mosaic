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
concrete `Engine` wired into the scheduler (`run.py`). The **extraction engine**
(downstream of storage) is also complete through Phase 3: Phase 0 (document-type
inference + validation), Phase 1 (chunking), Phase 2 (embedding), and Phase 3 (Chroma
write) are all in place; see "Extraction layer" below. Phase 4 (wiring into the
scheduler — hot path / cold path) is now wired too. Ingestion and extraction were built
as two parallel workstreams and merged — see "Document schema: two type fields" below
for the one contract point where that merge required a deliberate reconciliation, and
"Package naming" below for how the `processing/` vs `extraction/` naming split was
resolved. The **retrieval engine** (`retrieval/`, downstream of Chroma) is also built
end to end — all five phases of the Retrieval Pipeline spec (router → search → rerank →
cluster → output); see "Retrieval engine" below. It was originally built **ahead of**
Phase 2 (embeddings) and Phase 3 (Chroma write); both now exist (see above), but
retrieval hasn't yet been re-verified against a Chroma collection populated by a live
run of the real extraction pipeline — only against `FakeChromaCollection` fixtures and
an ephemeral collection the integration test seeds itself. See that section's
dependency-gap note before assuming it runs end to end today. The **generation engine**
(`generation/`, downstream of retrieval) is now also built end to end — all five phases
of the Generation Pipeline spec (prompt assembly → synthesis → claim parsing → citation
validation → output formatting); see "Generation engine" below. Unlike retrieval,
generation has no live-network dependency gap: every phase is fully offline-testable
(Gemini and the semantic-fallback embedder are both injectable), so its golden-path
integration test runs in CI rather than being skipped. The **query engine** (`query/`,
the read-path composition root) now ties retrieval and generation together into a single
`answer(query, profile) -> QueryResult` call, with a `query/run.py` CLI harness to drive
one question against the live Chroma collection; see "Query engine" below.

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
- `sentence-transformers`, `groq`, `chromadb` are real deps now (retrieval's Phase 1
  query embedding + classification, Phase 2 vector search). The first two are
  lazy-imported behind `processing/utils/embedding.py` / `retrieval/router.py` and
  injectable in tests, same pattern as `feedparser`/`transformers` — the offline suite
  needs none of the three installed.
- `google-genai` is a real dep too (generation's Phase 2 Gemini Flash call), lazy-imported
  behind `generation/synthesizer.py`'s `Synthesizer` and injectable in tests, same pattern.
  `GEMINI_API_KEY` (alongside `GROQ_API_KEY`) is the env var a live run needs; neither is
  read unless a caller omits the injectable client.

## Commands
```bash
python -m venv .venv && source .venv/bin/activate   # one-time setup
pip install -r requirements.txt                      # install dependencies
ruff check .                                         # lint
ruff format .                                        # format
pytest                                               # run tests

# Run the pipeline (write path, then inspect, then ask):
python -m ingestion.run  --once --chroma-path data/chroma   # fetch + normalize + (hot) extract
python -m extraction.run --once --chroma-path data/chroma   # cold-path backfill of unprocessed docs
python -m tools.inspect_pipeline                            # read-only stage-by-stage report
python -m query.run "your question here"                    # ask a question (read path end to end)
```
`query/run.py` degrades to whatever's configured: routing uses Groq when `GROQ_API_KEY` is
set (else an offline embedding+profile router, or force it with `--offline-router`), and
synthesis uses Gemini when `GEMINI_API_KEY` + `google-genai` are present (else it stops
after retrieval and prints the retrieved context). So a no-key run still exercises the whole
retrieval half live.

## Metrics
`Metrics.md` (repo root) is the running log of measured outcomes per phase — latency,
accuracy, cost, corroboration counts — feeding resume bullets and an ADR trail later.
Append real numbers there when a phase is benchmarked; write `not measured` rather than
guessing, and never invent precision. Most retrieval-engine rows are still `not measured`
as of this writing — the engine is built and unit-tested but has no live Chroma
collection or Groq key to benchmark against yet.

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
  scheduler that reads due-ness; the same scheduler now also drives hot-path extraction
  (see "Extraction layer" below) — no second scheduler.

## Data retention (planned, not yet implemented)
Local stores (`raw.db`, `data/chroma`) are meant to hold recent data only, not accumulate
indefinitely — stale documents should be automatically purged on a **tier-dependent
window, roughly 24-48 hours**. Exact per-tier thresholds and which direction (does a
higher-trust tier get a longer or shorter window?) are still open decisions — not yet
specified, don't guess a table here until they're pinned down. No deletion code exists
yet; this is written down now specifically so the decision survives until it's
implemented. Also flagged: local SQLite/Chroma storage itself may be superseded by a
server-backed store later — when that lands, this retention logic needs to move with it
(or be re-decided for whatever replaces `raw.db`), not be reimplemented twice.

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
  `transformers` (chunk-sizing tokenizer), `chromadb`, `sentence-transformers` (Phase 2/3
  embedding + vector store). `requirements-dev.txt` has `pytest`, `pytest-mock`, `ruff`.
  Adapters import `feedparser`/`requests` lazily inside their fetch methods, and
  `extraction/embedder.py`/`chroma_store.py` import `sentence_transformers`/`chromadb`
  lazily too, so the unit suite runs without any of them installed — only live-network,
  real-embedding, or integration runs need `requirements.txt`.

## Extraction layer
The extraction layer sits **downstream of storage**: it reads normalized Documents,
infers their type, chunks by type, embeds, and writes vectors to Chroma. It is a
**separate top-level package** (`extraction/`, sibling of `ingestion/`) that ingestion
and query-time retrieval both call into — never logic baked into ingestion.

**Folder rule:** ALL extraction logic lives under `extraction/`, even the parts that
run *during* the ingestion window (the hot path). Extraction logic is never split
between `ingestion/` and `extraction/`. This keeps the structure clean and avoids
merge conflicts between parallel workstreams.

### Package naming: `processing/` vs `extraction/`
This package was scaffolded twice under two names by parallel workstreams: an earlier
`processing/` (Phase 0–1 only) and a later `extraction/` that diverged from it — same
Phase 0–1 logic plus the Phase 2/3 orchestrator (`embedder.py`, `chroma_store.py`,
`extraction_engine.py`) and a CLI (`extraction/run.py`). The two also disagreed on the
`Chunk` schema (`identity_key` field, `published_date` type). **`extraction/` was kept
as canonical; `processing/` was deleted outright, not merged.** Every reference to a
"processing" package elsewhere in this file (or in code comments/docstrings you may
still encounter) means `extraction/` — treat `processing/` as a stale name if you see it.

**Two trigger paths**, driven by config, not hardcoded by tier:
- **Hot path** — inside the existing single scheduler (`ingestion/run.py`): a
  per-source `SourceConfig.processing_mode` field (`"hot"` | `"cold"`, default `"cold"`)
  decides whether `ConcreteEngine` extracts a document inline right after it's stored.
  `ConcreteEngine` never imports `extraction.*` directly — see "Hot path wiring" below
  for why — instead it calls an injected `on_processed` callback that `run.py` wires up
  to `extraction_engine.extract()`.
- **Cold path** — `extraction/cold_path.py`'s `ensure_processed(doc_id, ...)` processes
  a document on demand (e.g. a query-time cache-miss on Chroma) if it isn't already
  processed. It's a tested, ready-to-call function; nothing calls it yet because the
  `retrieval/` layer that would call it on a cache-miss doesn't exist yet (future work).
The hot/cold split is driven by **processing throughput + per-source cost vs the
ingestion window**, not by tier number (Tier 0/1 usually hot, Tier 3 usually cold,
Tier 2 depends on velocity) — `processing_mode` is set per source in `sources.yaml`,
not inferred from `tier`. One scheduler only — no second scheduler.

### Hot path wiring (`ingestion/engine.py` ↔ `extraction/`)
`tests/test_handoff.py`'s `TestNoDownstreamCoupling` pins a boundary: **`ingestion/`
must never import `extraction/` (or any downstream stage)** — the raw store is meant to
be the only seam, since the two halves can run on different clocks. Calling
`extraction_engine.extract()` straight from `ingestion/engine.py` would violate that.
Instead, `ConcreteEngine.__init__` takes an optional `on_processed: Callable[[Document],
None] | None` callback (untyped beyond `Callable` — no import of `extraction.*` in
`ingestion/`). `ingestion/run.py`'s `main()` — the composition root, not internal
ingestion logic — builds the actual closure that wires `MiniLMEmbedder` +
`ChromaVectorStore` + `extract()` and passes it in. This keeps the coupling boundary
real (verified by the test) while still letting hot-path sources embed inline.

**Phase map:** 0 type inference + validation · 1 chunking (paragraph/section/fixed) ·
2 embeddings (MiniLM/Gemini — embeddings only, never type detection) · 3 Chroma write ·
4 wire into the scheduler (hot path via `SourceConfig.processing_mode` + injected
callback; cold path via `extraction/cold_path.py`, not yet called by anything) ·
5 query-time fallback — retrieval layer calling `ensure_processed()` on a cache-miss
(not yet built).

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
- **Shared `extraction/text_metrics.py`** owns token/paragraph/marker counting so
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
  chunks with the real MiniLM tokenizer (`extraction/utils/tokenization.py`, lazy-loaded +
  cached; adds `transformers`), so window sizes align with the Phase 2 embedder. This is a
  *distinct* notion of "token" from `text_metrics.count_tokens` (the Phase-0 whitespace
  proxy) and is the one place a model appears before Phase 2 — chunk sizing only, never
  type detection (that stays heuristic). Chosen over the whitespace proxy on purpose.
- **Pure, no I/O.** Chunking takes in-memory Documents and returns Chunks. The offline
  test suite injects a word-level fake tokenizer (`tests/extraction/conftest.py`) so it
  never downloads MiniLM or imports `transformers`.
- **`Chunk` now carries `ordinal` and `section_label`.** Originally deferred as a Phase 1
  non-goal, then restored because the Retrieval Pipeline spec calls their absence a
  citation-metadata blocker. `ordinal` is stamped on every chunk; `section_label` is the
  detected header text for section chunks (including sub-chunks from an oversized
  section's fallback split) and stays `None` for paragraph/fixed chunks and header-less
  preambles — that's a legitimate content property, not missing data (see retrieval
  engine's `citation_fields_present` note below).

### Phase 2/3 decisions (embedding + Chroma write)
- **`extraction/embedder.py`**: `Embedder` is a `Protocol` (`model_name` + `embed()`);
  `MiniLMEmbedder` lazy-loads `sentence-transformers/all-MiniLM-L6-v2` on first call so
  import stays fast and the unit suite can inject a `FakeEmbedder` without the dependency
  installed. `model_name` is the canonical slug stamped into the Chroma collection name —
  enforces the one-model-per-collection rule from Conventions above.
- **`extraction/chroma_store.py`**: `ChromaVectorStore` wraps a `chromadb.ClientAPI`.
  Collection name encodes the embedder's `model_name`; writing with a mismatched model
  raises `ModelMismatchError` at write time. Collections are always created with cosine
  distance (matches MiniLM's L2-normalized vectors). On upsert, existing chunks sharing
  the new batch's `identity_key` are deleted first — this is the L2-update stale-chunk
  eviction so retrieval never surfaces a superseded document version.
- **`extraction/extraction_engine.py`**: `extract(documents, embedder, chroma_store,
  source_hints=...)` runs Phase 0→3 per document with per-document isolation (one bad
  document is caught, logged, and counted in `ExtractionResult.errors`; the rest
  continue). This is the one function both the hot path and cold path call — same
  function, different caller, per the Phase map above.

## Retrieval engine
`retrieval/` (sibling top-level package of `ingestion/` and `extraction/`) implements the
Retrieval Pipeline spec end to end: user query → ranked, clustered chunks → a typed
`RetrievalOutput` for the (not-yet-built) Generation Pipeline. Five phases, one module
each, all pure/injectable and unit-tested offline against fakes:

`router.py` (`QueryRouter`) → `search.py` (`VectorSearch`) → `rerank.py` (`Ranker`) →
`cluster.py` (`StoryClusterer`) → `output.py` (`assemble_retrieval_output`). Shared
dataclasses live in `contracts.py` (`RoutingResult`, `RetrievedChunk`, `StoryCluster`,
`UserProfile`).

**Locked decisions, as implemented:**
- **Metadata filter runs before ANN search.** `search.build_where_clause` combines
  `ticker $in` and `published_epoch $gte` with `$and`; returns `None` (not `{}`) when
  routing carries no constraints, since Chroma treats `where={}` as invalid.
- **Tier is a label, never a ranking lever.** `rerank.final_score` never references
  `RetrievedChunk.tier` — pinned by a test asserting a more-relevant Tier 3 chunk
  outranks a less-relevant Tier 1 one, the spec's own example.
- **Static, hand-tuned re-rank weights**, all named constants in `rerank.py`:
  `final_score = 0.5*relevance + 0.3*recency + 0.2*profile_bias`. `profile_bias` only
  applies its ticker-match term (`TICKER_MATCH_BIAS`) — the sector-match term
  (`SECTOR_MATCH_BIAS`) is a documented no-op, since `RetrievedChunk`'s field set has no
  sector to compare against.
- **L3 clustering reuses ingestion's dedup logic directly**, not a reimplementation:
  `cluster.py` imports `cosine_similarity` and `L3_SIMILARITY_THRESHOLD` from
  `ingestion.pipeline.dedup` (both made public there specifically for this reuse — they
  were `_cosine_similarity`/module-private before). **Note the spec-text discrepancy**:
  the Retrieval Pipeline spec's prose says "~0.92 cosine"; the actual, reused constant is
  `0.85`. Confirmed deliberately — reuse the real constant, not the approximation.
- **Query embedding uses the same model as the corpus.** `extraction/utils/embedding.py`
  is the one place that model (`sentence-transformers/all-MiniLM-L6-v2`, matching the
  Phase 1 chunk-sizing tokenizer) is named for single-string query embedding — lazy-loaded
  and cached, mirroring `extraction/utils/tokenization.py`. `router.py`'s default embedder
  calls through here. Note this is a distinct code path from the batch corpus embedder,
  `extraction/embedder.py`'s `MiniLMEmbedder` — both hardcode the same model ID
  independently rather than one delegating to the other; keep them in sync by hand if the
  model ever changes.

**Deviations from the spec's literal field lists** (both additive, both necessary —
documented here so they're not mistaken for drift):
- **`RetrievedChunk.embedding`** (optional, default `None`) isn't in the spec's Phase 2
  output code block, but Phase 4's own non-goal ("no new embedding model — reuse existing
  chunk vectors") presupposes chunk vectors are available somewhere between Phase 2 and
  Phase 4, and there's nowhere else for them to live. `VectorSearch` requests embeddings
  from Chroma explicitly (`include=[..., "embeddings"]` — omitted by default) and
  populates this field; a chunk with no embedding becomes its own singleton cluster in
  Phase 4 rather than raising.
- **`citation_fields_present` checks `ordinal` only, not `section_label`.** `ordinal` is
  stamped on every chunk by construction, so its absence is a real degradation signal.
  `section_label` is legitimately `None` for paragraph/fixed chunks by design (no section
  concept for articles/tweets) — requiring it universally would make the flag false for
  almost every non-filing-heavy result set, which isn't a useful signal.

**Dependency gap (read before assuming retrieval runs live):** Phase 2 (embeddings) and
Phase 3 (Chroma write) now exist (`extraction/embedder.py`, `extraction/chroma_store.py`
— see "Extraction layer" below), so the collection retrieval queries can, in principle,
be populated by a real ingest → extract run. What hasn't happened yet is proving that
end to end: retrieval is still only unit-tested against the `chromadb.Collection`
interface via `FakeChromaCollection` fixtures, and the one integration test
(`tests/retrieval/test_integration.py`) seeds its own ephemeral in-process Chroma
collection rather than one written by `extraction_engine.extract()`. It does need
network (MiniLM download, Groq API + `GROQ_API_KEY`) and is skipped by default, same
convention as the ingestion suite's live-network tests.

## Generation engine
`generation/` (sibling top-level package of `ingestion/`, `extraction/`, `retrieval/`)
implements the Generation Pipeline spec end to end: `RetrievalOutput` → prompt → Gemini
synthesis → parsed claims → grounding-validated claims → a typed `GeneratedAnswer` for the
user-facing surface. Five phases, one module each:

`prompt_builder.py` (`PromptBuilder`) → `synthesizer.py` (`Synthesizer`) →
`claim_parser.py` (`ClaimParser`) → `validator.py` (`CitationValidator`) →
`formatter.py` (`AnswerFormatter`). Shared dataclasses live in `contracts.py` (`LensDoc`,
`ParsedClaim`, `ValidatedClaim`, `Citation`, `GeneratedAnswer`).

**Locked decisions, as implemented:**
- **Structured output is the parsing strategy.** Gemini is prompted for exact
  `CLAIM:`/`SOURCE_CHUNK_ID:`/`CONFIDENCE:` blocks split by `---`; `claim_parser.py` is a
  deterministic line-prefix parser, never an LLM re-parse of freeform prose.
- **No model-emitted spans, ever.** `formatter.py`'s citation sentence selection is a
  plain word-overlap heuristic (Jaccard over punctuation-stripped, lowercased word sets)
  against `text_metrics.sentence_spans` — the same sentence-boundary definition Phase 1
  chunking's highlight-span selection uses (extracted there specifically for this reuse;
  see "Extraction layer" → Phase 1 and `extraction/utils/highlight.py`).
- **Tier is context, never a filter.** `formatter.py` cites the best-grounded claim
  regardless of its source's tier; Tier 2/3 sources get an inline skepticism note appended
  to the `Citation.source` label, never suppression.
- **Reject, don't repair.** An ungrounded claim is dropped, not sent back to Gemini for
  another round — no regenerate-on-failure loop anywhere in this pipeline.
- **Fail-closed synthesis.** `Synthesizer` retries a failed Gemini call with exponential
  backoff, then gives up and returns a marker string that is deliberately not valid
  CLAIM/SOURCE_CHUNK_ID/CONFIDENCE text — it flows through claim_parser/validator as zero
  grounded claims, landing on the same honest empty-state answer as any other
  fully-hallucinated response, with no separate failure-signaling path between phases.

**Scope-boundary deviation from the spec's phase organization (read before assuming
validator.py drops claims):** the spec's "reject, don't repair" *policy* — actually
dropping ungrounded claims, the >30%-dropped confidence warning, and the zero-survivor
honest empty state — is written under Phase 4's heading, but those are actions on the
*assembled answer*, which only Phase 5 builds. `validator.py`'s `CitationValidator`
therefore only makes the per-claim grounding decision (`is_grounded`, confidence,
supporting chunk) and returns every claim, grounded and ungrounded alike; `formatter.py`'s
`AnswerFormatter` is where dropping/warning/empty-state actually happens. Both modules'
docstrings cross-reference this split explicitly.

**Deviations from the spec's literal field/input lists** (both additive, both necessary —
documented here so they're not mistaken for drift, same convention as retrieval's):
- **`ParsedClaim.source_chunk_id`/`.confidence` are optional, plus a new `is_valid`
  flag** — the spec's literal 3-field `ParsedClaim` can't represent "this block had no
  usable ID" any other way, and Phase 3's own logic requires malformed/ID-less blocks to
  be "passed forward marked invalid," not silently dropped.
- **`AnswerFormatter.format` takes `clusters: list[StoryCluster]`** beyond the spec's
  literal Phase 5 input list (`validated_claims`, `chunks`) — needed to derive
  `corroboration_summary` from retrieval's already-computed `StoryCluster.outlet_count`
  instead of reinventing grouping logic in generation.
- **`corroboration_summary` is keyed by `StoryCluster.cluster_id`**, a real stable
  identifier, not a human-readable topic label like the spec's illustrative
  `"earnings_beat"` — generating an actual topic label would need summarization or an LLM
  call, out of scope for a phase whose entire purpose is that nothing here can hallucinate
  content.
- **`LensDoc` ships no bundled content.** It's a minimal `(title, text)` dataclass;
  real investing-framework content is product content, not something to invent while
  implementing architecture — `PromptBuilder` accepts whatever list it's given.

**No live-network dependency gap, unlike retrieval:** every generation phase is
fully offline-testable — the Gemini client and the semantic-fallback embedder are both
injectable, so `tests/generation/test_integration.py`'s golden-path test runs in CI
(not skipped). It does not need a live Chroma collection either: it starts from a
hand-built `RetrievalOutput` fixture, matching the spec's own "Consumes: RetrievalOutput
from the Retrieval Pipeline" framing.

## Query engine
`query/` (sibling top-level package) is the **read-path composition root** — what
`ingestion/run.py`'s `main()` is to the write path. `query/engine.py`'s
`answer(query, profile, *, collection, router, synthesizer=None, ...) -> QueryResult` is
the one place the query-time stages are wired in order: retrieval (router → search →
rerank → cluster → output) then generation (prompt → synthesize → parse → validate →
format). It's pure orchestration — the Chroma `collection`, `router`, and `synthesizer`
are all injected, never constructed here — so it's fully fake-testable
(`tests/query/test_engine.py`), same discipline as the phases it composes. This is the
call the future `interfaces/` layer makes; the UI shouldn't re-wire the nine stages.

**Locked decisions, as implemented:**
- **`answer()` returns `QueryResult(routing, retrieval, answer)`**, not the bare
  `GeneratedAnswer` the earlier report sketched — a justified richer return so callers
  (the CLI, later the UI's debug view) can see routing + retrieval, not only the final
  prose. `.answer` is the `GeneratedAnswer`; a UI that only wants prose reads that field.
- **`synthesizer=None` runs retrieval only** (`.answer` is None, `.routing`/`.retrieval`
  populated). This keeps all orchestration in one module: the CLI's no-Gemini path is a
  parameter, not a duplicated retrieval loop. It's also the graceful-degradation seam the
  harness leans on.
- **`OfflineRouter` / `route_offline` are the no-Groq fallback** (`query/engine.py`):
  embed the query with the shared MiniLM embedder, take tickers/sectors from the profile,
  leave `intent="unknown"`. Router-shaped so `answer()` treats it and the real
  `QueryRouter` identically. It cannot infer intent or extract tickers from query text —
  only semantic search + the profile's declared filter survive.
- **`query/run.py` is a read-only operator harness**, the counterpart to
  `tools/inspect_pipeline.py`. It never fetches/extracts/writes; it opens the existing
  Chroma collection, picks router (Groq vs offline) and synthesizer (Gemini vs none) from
  what's configured, prints routing + retrieval + the cited answer, and on a missing
  collection/key tells the operator exactly which command or env var to add.

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
  test_engine.py           # end-to-end: dedup branches, 304, transport rejection, drop-and-count,
                            # hot-path extraction call + status update
  test_handoff.py          # ingestion -> extraction boundary: raw store is the only seam,
                            # ingestion/ never imports extraction/
  extraction/               # extraction layer tests (type inference, validation, chunking,
                            # embedding, Chroma write, cold path)
  retrieval/               # retrieval layer tests (router, search, rerank, cluster, output)
    conftest.py               # FakeGroqClient, FakeChromaCollection, fake_query_embedder
    fixtures.py                # make_routing_result, make_retrieved_chunk, make_story_cluster
    test_integration.py       # skipped-by-default: full pipeline over a real Chroma collection
    test_*_adversarial.py     # edge-case/malformed-input coverage, one file per retrieval phase
  generation/              # generation layer tests (prompt, synthesis, parsing, validation, format)
    fixtures.py                # make_lens_doc, make_parsed_claim, make_validated_claim, ...
    test_integration.py       # golden-path: RetrievalOutput fixture -> GeneratedAnswer, runs in CI
  query/                   # query engine tests (read-path orchestration)
    test_engine.py            # answer() over fake collection/router/synthesizer; retrieval-only + full chain
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
