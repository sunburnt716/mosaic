"""
mosaic processing layer — downstream of storage, upstream of Chroma.

The processing layer reads normalized Documents and turns them into retrievable
vectors: infer type (Phase 0) -> chunk by type (Phase 1) -> embed (Phase 2) ->
write to Chroma (Phase 3). Ingestion (hot path) and query-time retrieval (cold path)
both call into this layer; ALL processing logic lives here, never split into the
ingestion package, so the two layers stay independently evolvable.

Phase 0 (this module set) is the foundation both paths share: algorithmic, model-free
document-type inference and structure validation. Public surface:

  infer_document_type(document, source_hints=None) -> str
  validate_document(document, inferred_type) -> ValidationResult
"""

from __future__ import annotations

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
    "DOCUMENT_TYPES",
    "FILING",
    "ARTICLE",
    "TWEET",
    "UNKNOWN",
]
