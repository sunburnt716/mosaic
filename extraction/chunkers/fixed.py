"""Fixed-size chunking — the fallback strategy for unstructured prose.

Use when text has no natural boundaries to exploit (generic articles, blogs, or any
document type without a registered strategy). Slides a fixed token window over the body
with configurable overlap so adjacent chunks share context at their seams.

Fixed chunks have no sub-structure to highlight, so `highlight_span == full_span` (the
whole window is the excerpt) — per the Phase 1 spec's dual-span rules.

`_plan_fixed` returns spans relative to the text it is given; the section chunker reuses
it to break an oversized section. `chunk_fixed` wraps it over the whole document body.
"""

from typing import TYPE_CHECKING

from extraction.chunk import Chunk, Span, materialize_chunks
from extraction.utils.tokenization import token_spans

if TYPE_CHECKING:
    from ingestion.core.document import Document

DEFAULT_CHUNK_SIZE = 512  # tokens per window
DEFAULT_OVERLAP = 50  # tokens shared between adjacent windows


def _plan_fixed(text: str, chunk_size: int, overlap: int) -> list[tuple[Span, Span]]:
    """Plan fixed windows over `text`, returning (full_span, highlight_span) per chunk.

    Spans are character offsets relative to `text`. For fixed chunks the highlight is the
    whole window, so both spans are identical.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    spans = token_spans(text)
    if not spans:
        return []

    step = chunk_size - overlap
    plans: list[tuple[Span, Span]] = []
    for start in range(0, len(spans), step):
        window = spans[start : start + chunk_size]
        full: Span = (window[0][0], window[-1][1])
        plans.append((full, full))  # highlight == full for fixed chunks
        if start + chunk_size >= len(spans):
            break  # this window reached the end; no trailing all-overlap chunk
    return plans


def chunk_fixed(
    document: "Document",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    chunked_at: str | None = None,
) -> list[Chunk]:
    """Chunk a document into fixed-size, overlapping token windows."""
    plans = _plan_fixed(document.body, chunk_size, overlap)
    return materialize_chunks(document, plans, chunked_at=chunked_at)
