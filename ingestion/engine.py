# Orchestrator: wires every pipeline stage together and drives the ingestion run.
#
# The engine is the only place where side effects (fetch, store, seen-store update) are
# coordinated. All pipeline stages (normalizer, dedup classifier) are pure; the engine
# decides when and whether their results are committed.
#
# High-level flow for each source in sources.yaml:
#   1. Load SourceConfig and resolve the adapter via adapters/registry.py.
#   2. Call adapter.fetch(config) to get an iterable of raw item dicts.
#   3. For each raw item:
#        a. normalizer.normalize(raw, config, fetched_at)  ->  Document
#        b. dedup.classify(doc, seen_store)                ->  DedupResult
#        c. Branch on DedupResult:
#             NEW          -> save_raw + save_document + set_hash
#             L1_DUPLICATE -> discard (no writes)
#             L2_UPDATE    -> save_raw + save_document (overwrite) + set_hash
#             L3_NEAR_DUP  -> save_raw + save_document + set_hash (tag cluster_id)
#   4. Log per-source summary: fetched / new / l1 / l2 / l3 counts.
#
# Source isolation: a failure in one source (FetchError, NormalizationError) must be caught
# and logged without aborting processing of remaining sources. Each source runs in its own
# error boundary so one bad feed can't poison the run.
#
# Concurrency: sources are independent and may be fetched concurrently (e.g. via asyncio or
# a thread pool). The engine controls concurrency; adapters and pipeline stages must be
# stateless/thread-safe.
#
# The engine does NOT chunk, embed, or write to Chroma — those are the next pipeline stage
# downstream and are out of scope for the ingestion engine.
