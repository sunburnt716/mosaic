"""
Defines the canonical Document schema — the central contract for the entire pipeline.

Every upstream adapter produces raw data in its own shape; every downstream stage
(normalizer, dedup, raw store, chunker, embedder, Chroma) consumes Documents. This
schema is the single point where those two halves meet.

A plain, mutable dataclass is deliberate: validation of required fields lives in the
normalizer (which raises NormalizationError), keeping this schema dependency-free. Later
stages (enrichment, status transitions) update fields on the same instance rather than
rebuilding it.

Field groups
------------
Ingest-time fields (set by the normalizer):
  - id              : globally unique document identifier (derived in hashing.py).
  - content_hash    : SHA-256 of the normalized content bytes; drives L1 exact dedup.
  - identity_key    : stable key for the same logical article across updates
                      (source + article id); drives L2 same-article-updated dedup.
  - source_name     : human-readable name of the originating source (e.g. "Reuters").
  - url             : canonical URL of the article or filing; must survive into Chroma.
  - tier            : trust level (0–3) stamped at ingest from the source config —
                      never inferred later.
  - published_date  : timestamp of original publication; drives recency ranking.
  - title           : headline or filing subject line.
  - body            : full plain-text content.
  - doc_type        : the human-authored advisory hint from SourceConfig.doc_type
                      ("article" | "filing"), stamped verbatim at ingest. This is
                      NOT the type chunking dispatches on — see document_type below.
  - raw_payload     : the original, untouched response from the source; preserved so
                      downstream stages can re-run offline without re-fetching.
  - fetched_at      : UTC timestamp when this document was retrieved.

Enrichment fields (populated by later stages; empty at ingest):
  - tickers, sectors, key_points

Pipeline state:
  - status : "unprocessed" is the ingestion handoff state — ingestion's only output is
    unprocessed Documents in the raw store. The processing stage reads them on its own
    clock and advances this (e.g. -> "processed"). Ingestion never sets it to anything
    else; downstream owns every later transition.

Processing-populated fields (set by the processing layer, downstream of storage):
  - document_type      : the structure-inferred type ("filing" | "article" | "tweet"
                         | "unknown") produced by processing Phase 0. Optional and
                         informational: None until inference runs. This is the
                         AUTHORITATIVE per-document type that Phase 1 chunking dispatches
                         on. It is DISTINCT from doc_type above (the advisory hint):
                         doc_type's vocabulary is only article/filing and is never
                         inferred, so it cannot express "tweet"; inference may consult
                         doc_type and override it. Two fields for two purposes — the
                         chunking registry keys on document_type, never doc_type.
  - validation_warnings: structured warnings emitted by processing validation when a
                         document's structure disagrees with its document_type (e.g. a
                         filing with no section headers). Informational only — recorded
                         so source quality can be monitored over time.

Constraints
-----------
  - source_name, url, tier, and published_date MUST be non-null so synthesis can
    always cite + timestamp.
  - raw_payload must never be modified after ingest; treat it as append-only.
  - document_type and validation_warnings are the deliberate, documented exception to
    "derive at ingest, never infer from content": they are inferred from content by the
    processing layer, are purely informational, and never gate citation or dedup.
    Citation and dedup depend only on ingest-time fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Document:
    # --- identity (derived in hashing.py) ---
    id: str
    content_hash: str
    identity_key: str
    # --- citation metadata (must survive into Chroma) ---
    source_name: str
    url: str
    tier: int
    published_date: datetime
    # --- content ---
    title: str
    body: str
    doc_type: str  # advisory hint stamped from SourceConfig.doc_type; see module docstring
    # --- provenance ---
    raw_payload: Any  # untouched original; preserved for offline replay
    fetched_at: datetime
    # --- enrichment (populated by later stages; empty at ingest) ---
    tickers: list = field(default_factory=list)
    sectors: list = field(default_factory=list)
    key_points: list = field(default_factory=list)
    # --- pipeline state ---
    status: str = "unprocessed"
    # --- processing-populated (optional, informational; see module docstring) ---
    document_type: Optional[str] = None
    validation_warnings: list[str] = field(default_factory=list)
