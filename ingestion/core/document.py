"""
Defines the canonical Document schema — the central contract for the entire pipeline.

Every upstream adapter produces raw data in its own shape; every downstream stage
(normalizer, dedup, raw store, chunker, embedder, Chroma) consumes Documents. This
schema is the single point where those two halves meet, so it is deliberately small,
typed, and immutable.

Field groups
------------
Ingest-time fields (set by the normalizer, never changed afterwards):
  - id              : globally unique document identifier (derived in hashing.py).
  - content_hash    : SHA-256 of the normalized content bytes; drives L1 exact dedup.
  - identity_key    : stable key for the same logical article across updates
                      (source + article id); drives L2 same-article-updated dedup.
  - source_name     : human-readable name of the originating source (e.g. "Reuters").
  - url             : canonical URL of the article or filing; must survive into Chroma.
  - tier            : trust level (0–3) stamped at ingest from the source config —
                      never inferred later.
  - published_date  : ISO-8601 timestamp of original publication; drives recency ranking.
  - title           : headline or filing subject line.
  - body            : full plain-text content; chunked downstream by document_type.
  - raw_payload     : the original, untouched response from the source; preserved so
                      downstream stages can re-run offline without re-fetching.
  - fetched_at      : UTC ISO-8601 timestamp when this document was retrieved.

Processing-populated fields (set by the processing layer, downstream of storage):
  - document_type      : the structure-inferred type ("filing" | "article" | "tweet"
                         | "unknown") produced by processing Phase 0. Optional and
                         informational: None until inference runs. This is the
                         authoritative per-document type that selects a chunking
                         strategy in Phase 1. It is DISTINCT from SourceConfig.doc_type,
                         which is only the human-authored advisory hint that inference
                         may consult and override.
  - validation_warnings: structured warnings emitted by processing validation when a
                         document's structure disagrees with its document_type (e.g. a
                         filing with no section headers). Informational only — recorded
                         so source quality can be monitored over time.

Constraints
-----------
  - source_name, url, tier, and published_date MUST be non-null so synthesis can
    always cite + timestamp.
  - raw_payload must never be modified after ingest; treat it as append-only.
  - Ingest-time fields are derived at ingest, never inferred from content. The two
    processing-populated fields above are the deliberate, documented exception: they
    are inferred from content by the processing layer, are purely informational, and
    never gate citation or dedup. Citation and dedup depend only on ingest-time fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Document:
    """Immutable, canonical representation of one ingested item.

    Frozen so that ingest-time fields cannot drift after the normalizer produces
    them. The processing layer does not mutate Documents in place; it computes
    document_type / validation_warnings as values and (in a later phase) rebuilds
    the Document via dataclasses.replace at the point where they are attached.
    """

    # --- Identity (derived in hashing.py) -----------------------------------
    id: str
    content_hash: str
    identity_key: str

    # --- Citation metadata (must survive into Chroma; never null) -----------
    source_name: str
    url: str
    tier: int
    published_date: str

    # --- Content -----------------------------------------------------------
    title: str
    body: str

    # --- Provenance --------------------------------------------------------
    raw_payload: dict[str, Any]
    fetched_at: str

    # --- Processing-populated (optional, informational) --------------------
    # Defaulted so a freshly-normalized Document is valid before processing runs.
    document_type: Optional[str] = None
    validation_warnings: list[str] = field(default_factory=list)
