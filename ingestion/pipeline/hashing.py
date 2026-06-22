# Pure functions for computing the three identifiers that every Document must carry.
#
# All functions here are deterministic and side-effect-free. They take raw adapter output
# and return stable, reproducible values. No I/O, no state.
#
# Functions:
#
#   content_hash(raw_body: str | bytes) -> str
#     SHA-256 of the normalized content bytes (UTF-8 encoded, whitespace-collapsed).
#     Used for L1 exact dedup: two documents with identical content_hash are byte-level duplicates.
#     Normalize before hashing (strip leading/trailing whitespace, collapse internal runs)
#     so that trivial formatting differences don't produce spurious distinct hashes.
#
#   identity_key(source_name: str, source_article_id: str) -> str
#     A stable, human-readable key that identifies the same logical article across updates.
#     Composed as "{source_name}::{source_article_id}" where source_article_id is whatever
#     the source uses as its own primary key (GUID in RSS, accession number in EDGAR, etc.).
#     Used for L2 same-article-updated dedup: if we've seen this key before with a different
#     content_hash, the article has been updated, not duplicated.
#
#   document_id(identity_key: str, content_hash: str) -> str
#     The globally unique Document.id, derived as a hash of (identity_key + content_hash).
#     Stable for the lifetime of a specific version of an article; changes if content changes.
#     Format: hex string, no dashes, lowercase — safe for use as a Chroma document ID.
