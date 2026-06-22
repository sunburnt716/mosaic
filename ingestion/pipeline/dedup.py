# Pure classification stage: decides, for each incoming Document, which dedup level it triggers.
#
# Dedup has three distinct levels that must NEVER be collapsed into one — each catches a
# different failure mode and requires a different response from the engine.
#
# classify(doc: Document, seen_store: SeenStore) -> DedupResult
#   Returns one of: NEW | L1_DUPLICATE | L2_UPDATE | L3_NEAR_DUPLICATE
#
# Level definitions:
#
#   L1 — Exact duplicate (content hash match)
#     doc.content_hash is already in seen_store.
#     The document is byte-for-byte identical to one we've already ingested.
#     Action: discard silently. Nothing new to store or embed.
#
#   L2 — Same article, updated (identity key match, different content hash)
#     doc.identity_key is in seen_store, but with a different content_hash.
#     The source has updated an article we already have (correction, expansion, etc.).
#     Action: store the new version; the engine may choose to re-embed and replace
#     the previous Chroma entry. The old raw_payload should be retained for audit.
#
#   L3 — Near-duplicate / same story, different outlet (embedding similarity)
#     No identity_key match, but embedding similarity to an existing document exceeds threshold.
#     Two outlets are covering the same underlying event.
#     Action: ingest BOTH — do not discard. L3 preserves cross-outlet corroboration so
#     retrieval can surface the same story from multiple sources for trust assessment.
#     Tag the document with a cluster_id linking it to its near-duplicates.
#
#   NEW — No match at any level.
#     Action: ingest normally.
#
# This function is pure with respect to classification logic. The SeenStore is read-only here;
# the engine is responsible for updating the store after a decision is made.
# L3 embedding comparison uses pre-computed embeddings — this module does not call any model.
