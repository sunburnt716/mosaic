# extraction/ (scaffold — not yet implemented)

The processing stage that sits **after** ingestion in the pipeline:

```
ingestion → [extraction] → generation
```

## Responsibility (future)
- Chunk Documents by `doc_type` (filings by section, articles by paragraph).
- Embed chunks (MiniLM local or Gemini) into Chroma — one embedding model per collection.
- Enrich: `tickers`, `sectors`, `key_points`.

## Handoff contract (the only coupling to ingestion)
Ingestion's sole output is `status: "unprocessed"` `Document` rows in the raw store
(`ingestion/storage/raw_store.py`). Extraction reads from that store **on its own clock** —
it does not call into the ingestion engine, and ingestion does not call into extraction.
After processing a Document, extraction advances its `status` (e.g. `processed`).

## Non-goals for now
Nothing here is built yet. The ingestion engine stops at the `unprocessed` handoff;
this folder is a placeholder so the pipeline structure (sources → adapters → normalizer →
dedup → raw store → **extraction** → generation) is visible in the layout.
