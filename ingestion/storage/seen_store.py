# Side-effecting store: tracks which documents have already been ingested to power L1/L2 dedup.
#
# The seen store is the lightweight lookup table that dedup.py reads to classify incoming documents.
# It records the last-seen content_hash for every identity_key the engine has processed.
#
# Responsibilities:
#   get_hash(identity_key: str) -> str | None
#     Return the content_hash last stored for this identity_key, or None if never seen.
#     Called by dedup.py to determine L1 (hash match) vs L2 (key match, hash differs) vs NEW.
#
#   set_hash(identity_key: str, content_hash: str) -> None
#     Record or update the content_hash for an identity_key after a document has been ingested.
#     Called by the engine after dedup classification, not by dedup.py itself (dedup is read-only).
#
#   contains_hash(content_hash: str) -> bool
#     Fast bloom-filter-style check: has this exact content_hash been stored for ANY identity_key?
#     Used as the L1 short-circuit before the more expensive identity_key lookup.
#
# Storage backend: same pluggable pattern as raw_store.py (SQLite for dev, Redis or Postgres
# for high-throughput production). The interface above is stable across backends.
#
# Consistency contract: set_hash must be called atomically with the raw_store.save_* calls
# inside the engine's transaction boundary. A partial write (document saved but hash not recorded)
# would cause the engine to re-ingest the same document on the next run.
