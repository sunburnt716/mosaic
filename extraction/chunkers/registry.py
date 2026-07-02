"""Strategy dispatcher — maps a Document's type to its chunking function.

Config-over-code: the strategy for a document follows from its `doc_type` (stamped at
ingest from source config), not from per-source branches. The mapping uses the doc_type
*values the pipeline actually produces* — "article" and "filing" (see the normalizer and
config/sources.json) — rather than aspirational type names:

  article  -> chunk_paragraph   (CLAUDE.md: "articles by paragraph")
  filing   -> chunk_section     (CLAUDE.md: "filings by section")
  <other>  -> chunk_fixed        (fallback for unstructured / unknown types)

Fixed-size is the deliberate fallback for any unmapped type, per the Phase 1 spec's
"Other unstructured text -> Fixed-size (fallback)".
"""

from typing import TYPE_CHECKING, Callable

from extraction.chunkers.fixed import chunk_fixed
from extraction.chunkers.paragraph import chunk_paragraph
from extraction.chunkers.section import chunk_section

if TYPE_CHECKING:
    from extraction.chunk import Chunk

Chunker = Callable[..., "list[Chunk]"]

_STRATEGY_BY_DOC_TYPE: dict[str, Chunker] = {
    "article": chunk_paragraph,
    "filing": chunk_section,
}

# Fallback for any doc_type without a registered strategy (unstructured text).
_FALLBACK: Chunker = chunk_fixed


def get_chunker(doc_type: str) -> Chunker:
    """Return the chunking function for a document type, falling back to fixed-size."""
    return _STRATEGY_BY_DOC_TYPE.get(doc_type, _FALLBACK)
