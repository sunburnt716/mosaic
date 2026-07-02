# extraction/

The processing stage that sits **after** ingestion in the pipeline:

```
ingestion → [extraction] → generation
```

## Responsibility
- **Chunk** Documents by `doc_type` (filings by section, articles by paragraph). *(built — Phase 1)*
- **Embed** chunks (MiniLM local or Gemini) into Chroma — one embedding model per collection. *(future)*
- **Enrich**: `tickers`, `sectors`, `key_points`. *(future)*

## Phase 1 — Chunking (built)

Converts normalized `Document`s into `Chunk`s ready for embedding. Pure functions (no I/O)
plus shared utilities; strategy is dispatched by document type.

```
extraction/
├── chunk.py                  # Chunk dataclass (the contract) + build/materialize helpers
├── utils/
│   ├── tokenization.py       # shared MiniLM tokenizer (lazy-loaded, cached) — one source of truth
│   ├── section_detection.py  # regex header detection (LLM-swappable, signature fixed)
│   └── highlight.py          # highlight-span (first-sentence) selection
├── chunkers/
│   ├── registry.py           # get_chunker(doc_type) → strategy
│   ├── fixed.py              # fixed-size token windows (fallback strategy)
│   ├── paragraph.py          # paragraph-grain chunking (articles)
│   └── section.py            # section-grain chunking (filings)
└── engine.py                 # orchestrator: chunk_document / chunk_documents
```

**Strategy dispatch** (by the `doc_type` values the pipeline actually produces):

| `doc_type` | strategy         | why |
|------------|------------------|-----|
| `article`  | `chunk_paragraph`| articles carry meaning at paragraph grain |
| `filing`   | `chunk_section`  | filings are organized under section headers |
| *other*    | `chunk_fixed`    | fallback for unstructured / unknown text |

**Dual spans** are load-bearing: `full_span` (whole chunk) feeds embedding/retrieval;
`highlight_span` (surgical excerpt) is what generation cites back to the user. Every Chunk
carries citation provenance (`source_name`, `url`, `tier`, `published_date`) copied from its
parent Document so it survives into Chroma.

**Tokenization lives only in `utils/tokenization.py`.** Chunkers size and slice by tokens
via those helpers; none loads a tokenizer of its own (mixing models in a collection is a
RAG-fitness failure). The MiniLM tokenizer is imported lazily and cached, so the offline unit
suite (which injects a fake tokenizer) never pulls in `transformers`.

## Handoff contract (the only coupling to ingestion)
Ingestion's sole output is `status: "unprocessed"` `Document` rows in the raw store
(`ingestion/storage/raw_store.py`). Extraction reads from that store **on its own clock** —
it does not call into the ingestion engine, and ingestion does not call into extraction.
Advancing `status` after processing belongs with the (future) embedding stage that persists
results; Phase 1 stops at producing `Chunk`s in memory.

## Non-goals (Phase 1)
Semantic/embedding-based boundaries, model-ranked highlight selection, a formal section-label
field on `Chunk`, and multilingual tokenization are all future work.
