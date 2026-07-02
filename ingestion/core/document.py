"""The canonical Document schema — the central contract for the entire pipeline.

Every upstream adapter produces raw data in its own shape; every downstream stage
(normalizer, dedup, raw store, chunker, embedder, Chroma) consumes Documents.

Constraints (enforced at construction by the normalizer, not here):
  - source_name, url, tier, and published_date MUST be non-null so synthesis can
    always cite + timestamp.
  - raw_payload must never be modified after ingest; treat it as append-only.

A plain dataclass is deliberate: validation of required fields lives in the
normalizer (which raises NormalizationError), keeping this schema dependency-free.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
    doc_type: str  # "article" (chunk by paragraph) | "filing" (chunk by section)
    # --- provenance ---
    raw_payload: Any  # untouched original; preserved for offline replay
    fetched_at: datetime
    # --- enrichment (populated by later stages; empty at ingest) ---
    tickers: list = field(default_factory=list)
    sectors: list = field(default_factory=list)
    key_points: list = field(default_factory=list)
    # --- pipeline state ---
    # "unprocessed" is the ingestion handoff state: ingestion's only output is
    # unprocessed Documents in the raw store. The (future) processing stage reads them
    # on its own clock and advances this (e.g. -> "processed"). Ingestion never sets it
    # to anything else; downstream owns every later transition.
    status: str = "unprocessed"
