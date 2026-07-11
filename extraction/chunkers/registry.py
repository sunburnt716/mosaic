"""
Strategy dispatcher — maps a Document's inferred type to its chunking function.

Config-over-code: the strategy follows from `Document.document_type` (the structure-inferred
type from Phase 0), not from per-source branches. The mapping uses the type constants defined
in `processing.type_inference`, so the chunking vocabulary can never drift from inference's:

  ARTICLE  -> chunk_paragraph   (CLAUDE.md: "articles by paragraph")
  FILING   -> chunk_section     (CLAUDE.md: "filings by section")
  <other>  -> chunk_fixed        (fallback: TWEET, UNKNOWN, and None-before-inference)

Fixed-size is the deliberate fallback for tweets (tiny → a single window), "unknown" documents,
and any document whose type has not been inferred yet, per the Phase 1 spec's "Other → fixed".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from extraction.chunkers.fixed import chunk_fixed
from extraction.chunkers.paragraph import chunk_paragraph
from extraction.chunkers.section import chunk_section
from extraction.type_inference import ARTICLE, FILING

if TYPE_CHECKING:
    from extraction.chunk import Chunk

Chunker = Callable[..., "list[Chunk]"]

_STRATEGY_BY_DOCUMENT_TYPE: dict[str, Chunker] = {
    ARTICLE: chunk_paragraph,
    FILING: chunk_section,
}

# Fallback for any type without a registered strategy (tweet, unknown, or not-yet-inferred).
_FALLBACK: Chunker = chunk_fixed


def get_chunker(document_type: Optional[str]) -> Chunker:
    """Return the chunking function for a document type, falling back to fixed-size."""
    return _STRATEGY_BY_DOCUMENT_TYPE.get(document_type, _FALLBACK)
