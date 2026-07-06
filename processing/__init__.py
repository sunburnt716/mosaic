"""
mosaic processing layer — downstream of storage, upstream of Chroma.

The processing layer reads normalized Documents and turns them into retrievable
vectors: infer type (Phase 0) -> chunk by type (Phase 1) -> embed (Phase 2) ->
write to Chroma (Phase 3). Ingestion (hot path) and query-time retrieval (cold path)
both call into this layer; ALL processing logic lives here, never split into the
ingestion package, so the two layers stay independently evolvable.

Phase 0 (type inference + validation) is the foundation both paths share: algorithmic,
model-free. Phase 1 (chunking) turns typed Documents into Chunks by strategy. Public surface:

  infer_document_type(document, source_hints=None) -> str        (Phase 0)
  validate_document(document, inferred_type) -> ValidationResult (Phase 0)
  chunk_document(document) -> list[Chunk]                        (Phase 1)
  chunk_documents(documents) -> list[Chunk]                      (Phase 1)
"""

from __future__ import annotations

from processing.chunk import Chunk
from processing.engine import chunk_document, chunk_documents
from processing.type_inference import (
    ARTICLE,
    DOCUMENT_TYPES,
    FILING,
    TWEET,
    UNKNOWN,
    infer_document_type,
)
from processing.validation import ValidationResult, validate_document

__all__ = [
    "infer_document_type",
    "validate_document",
    "ValidationResult",
    "chunk_document",
    "chunk_documents",
    "Chunk",
    "DOCUMENT_TYPES",
    "FILING",
    "ARTICLE",
    "TWEET",
    "UNKNOWN",
]
