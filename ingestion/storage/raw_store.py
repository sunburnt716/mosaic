# Side-effecting store: persists both raw payloads and normalized Documents for durable replay.
#
# The raw store is the append-only source of truth for everything the ingestion engine has fetched.
# Downstream stages (chunker, embedder, Chroma writer) can re-run from this store without
# re-fetching from external sources, making the pipeline resilient to downstream failures.
#
# Responsibilities:
#   save_raw(doc_id: str, raw_payload: dict | str) -> None
#     Persist the untouched adapter output for a document. Never modify raw_payload after write.
#     Keyed by doc_id; idempotent — re-saving the same doc_id must not corrupt existing data.
#
#   save_document(doc: Document) -> None
#     Persist the normalized, validated Document (without raw_payload to avoid duplication).
#     Keyed by doc.id. Overwrites on L2 update; must not be called for L1 duplicates.
#
#   get_document(doc_id: str) -> Document | None
#     Retrieve a previously stored Document by ID. Returns None if not found.
#
#   get_raw(doc_id: str) -> dict | str | None
#     Retrieve the raw payload for a document by ID. Returns None if not found.
#
# Storage backend is pluggable (SQLite for dev, Postgres or object store for prod).
# The interface above must remain stable regardless of backend — callers never touch
# the backend directly.
#
# Append-only invariant: raw payloads written once must never be mutated or deleted
# by the ingestion engine. Retention policy (archival, TTL) is an ops concern, not a code concern.
