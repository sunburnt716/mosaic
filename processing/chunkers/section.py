"""
Section-based chunking — the strategy for structured filings and regulatory docs.

Filings are organized under explicit headers ("RISK FACTORS", "Item 1A."), and their sections
are the meaningful retrieval unit (CLAUDE.md: "filings by section"). This splits the body at
detected headers (via processing/utils/section_detection.py, which reuses the same filing-marker
vocabulary Phase 0 types with) and emits one chunk per section — except a section too large to
embed well, which is recursively broken with a fallback strategy (paragraph by default, fixed
on request).

`highlight_span` is the first sentence *after* the header line (the Phase 1 heuristic). Section
labels are noted here but are not yet a formal Chunk field — a documented Phase 1 non-goal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from processing.chunk import Chunk, Span, materialize_chunks
from processing.chunkers.fixed import _plan_fixed
from processing.chunkers.paragraph import _plan_paragraph
from processing.utils.highlight import select_highlight_span
from processing.utils.section_detection import detect_section_headers
from processing.utils.tokenization import token_spans

if TYPE_CHECKING:
    from ingestion.core.document import Document

DEFAULT_MAX_SECTION_TOKENS = 1024  # break sections larger than this with the fallback
DEFAULT_FALLBACK_STRATEGY = "paragraph"  # "paragraph" | "fixed"

# A section is (start, end, content_start): body bounds plus where content begins after the
# header line (None for a header-less preamble, where content starts at `start`).
_Section = tuple[int, int, "int | None"]


def _split_sections(text: str) -> list[_Section]:
    """Split `text` into sections at detected headers, preserving header/content bounds."""
    header_end_by_start = {start: end for start, end in detect_section_headers(text)}
    if not header_end_by_start:
        return [(0, len(text), None)]  # no headers: the whole body is one section

    starts = sorted(header_end_by_start)
    sections: list[_Section] = []
    # Content before the first header is a header-less preamble section.
    if starts[0] > 0:
        sections.append((0, starts[0], None))
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        sections.append((start, end, header_end_by_start[start]))
    return sections


def _fallback_plans(strategy: str, text: str, max_section_tokens: int) -> list[tuple[Span, Span]]:
    """Break an oversized section using the configured fallback strategy."""
    if strategy == "fixed":
        return _plan_fixed(text, chunk_size=512, overlap=50)
    return _plan_paragraph(text, min_paragraph_tokens=50, max_paragraph_tokens=max_section_tokens)


def chunk_section(
    document: "Document",
    max_section_tokens: int = DEFAULT_MAX_SECTION_TOKENS,
    fallback_strategy: str = DEFAULT_FALLBACK_STRATEGY,
    chunked_at: str | None = None,
) -> list[Chunk]:
    """Chunk a document by section, recursively splitting sections that are too large."""
    body = document.body
    if not body:
        return []  # no content to section
    chunks: list[Chunk] = []
    ordinal = 0
    for start, end, content_start in _split_sections(body):
        segment = body[start:end]
        if len(token_spans(segment)) > max_section_tokens:
            plans = _fallback_plans(fallback_strategy, segment, max_section_tokens)
        else:
            full: Span = (0, len(segment))
            # Highlight the first sentence after the header line, not the header itself.
            hl_from = (content_start - start) if content_start is not None else 0
            plans = [(full, select_highlight_span(segment, start=hl_from))]
        new = materialize_chunks(
            document, plans, base=start, start_ordinal=ordinal, chunked_at=chunked_at
        )
        chunks.extend(new)
        ordinal += len(new)
    return chunks
