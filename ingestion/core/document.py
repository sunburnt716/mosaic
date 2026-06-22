# Defines the canonical Document schema — the central contract for the entire pipeline.
#
# Every upstream adapter produces raw data in its own shape; every downstream stage
# (normalizer, dedup, raw store, chunker, embedder, Chroma) consumes Documents.
# This schema is the single point where those two halves meet.
#
# Fields required on every Document:
#   - id              : globally unique document identifier (derived in hashing.py)
#   - content_hash    : SHA-256 of the raw content bytes; drives L1 exact dedup
#   - identity_key    : stable key for the same logical article across updates (source + article id);
#                       drives L2 same-article-updated dedup
#   - source_name     : human-readable name of the originating source (e.g. "Reuters", "SEC EDGAR")
#   - url             : canonical URL of the article or filing; must survive into Chroma for citation
#   - tier            : trust level (0–3) stamped at ingest from the source config — never inferred later
#   - published_date  : ISO-8601 timestamp of original publication; drives recency ranking downstream
#   - title           : headline or filing subject line
#   - body            : full plain-text content; chunked downstream by doc_type
#   - doc_type        : discriminates chunking strategy — "article" (by paragraph) vs "filing" (by section)
#   - raw_payload     : the original, untouched response from the source; preserved so downstream
#                       stages can re-run offline without re-fetching
#   - fetched_at      : UTC timestamp when this document was retrieved by the ingestion engine
#
# Constraints:
#   - source_name, url, tier, and published_date MUST be non-null so synthesis can always cite + timestamp.
#   - raw_payload must never be modified after ingest; treat it as append-only.
#   - Do not add fields that would need to be inferred or computed from content — derive them at ingest.
