"""Paragraph-based chunking — the strategy for news articles and long-form prose.

Articles carry meaning at the paragraph grain, so this splits on paragraph boundaries
rather than arbitrary token windows (per CLAUDE.md's RAG-fitness rule: "articles by
paragraph"). Very small paragraphs are merged forward into their neighbours so retrieval
isn't polluted by orphan one-liners.

Paragraph boundaries: the normalizer emits one `\n`-separated line per source paragraph
(it "preserves paragraph boundaries as newlines"), so a run of non-newline characters is
one paragraph. Splitting on newline runs also handles raw `\n\n`-separated text unchanged.

`highlight_span` is the paragraph's first sentence (the Phase 1 heuristic; model-ranked
selection is Phase 2+). `_plan_paragraph` returns spans relative to the text it is given so
the section chunker can reuse it on an oversized section.
"""

import logging
import re
from typing import TYPE_CHECKING

from extraction.chunk import Chunk, Span, materialize_chunks
from extraction.utils.highlight import select_highlight_span
from extraction.utils.tokenization import count_tokens

if TYPE_CHECKING:
    from ingestion.core.document import Document

logger = logging.getLogger(__name__)

DEFAULT_MIN_PARAGRAPH_TOKENS = 50  # merge paragraphs smaller than this into a neighbour
DEFAULT_MAX_PARAGRAPH_TOKENS = 1024  # warn (don't split) when a paragraph exceeds this

# A paragraph is a run of non-newline characters; blank lines fall between matches.
_PARAGRAPH_RE = re.compile(r"[^\n]+")


def _merge_small(text: str, paragraphs: list[Span], min_tokens: int) -> list[Span]:
    """Merge paragraphs under `min_tokens` forward into following ones (then the tail back).

    Spans are absolute into `text`; merging just extends the end offset, so the slice
    naturally re-includes the newline separators between the merged lines.
    """
    merged: list[Span] = []
    i = 0
    while i < len(paragraphs):
        start = paragraphs[i][0]
        end = paragraphs[i][1]
        # Grow this chunk until it clears the floor or we run out of paragraphs.
        while count_tokens(text[start:end]) < min_tokens and i + 1 < len(paragraphs):
            i += 1
            end = paragraphs[i][1]
        merged.append((start, end))
        i += 1

    # A too-small final chunk can't merge forward; fold it back into the previous one.
    if len(merged) >= 2 and count_tokens(text[merged[-1][0] : merged[-1][1]]) < min_tokens:
        last = merged.pop()
        merged[-1] = (merged[-1][0], last[1])
    return merged


def _plan_paragraph(
    text: str, min_paragraph_tokens: int, max_paragraph_tokens: int
) -> list[tuple[Span, Span]]:
    """Plan paragraph chunks over `text`, returning (full_span, highlight_span) per chunk."""
    paragraphs = [(m.start(), m.end()) for m in _PARAGRAPH_RE.finditer(text)]
    if not paragraphs:
        return []

    plans: list[tuple[Span, Span]] = []
    for start, end in _merge_small(text, paragraphs, min_paragraph_tokens):
        segment = text[start:end]
        if count_tokens(segment) > max_paragraph_tokens:
            logger.warning(
                "paragraph chunk of %d tokens exceeds max_paragraph_tokens=%d "
                "(kept whole; consider section/fixed chunking)",
                count_tokens(segment),
                max_paragraph_tokens,
            )
        hl_start, hl_end = select_highlight_span(segment)
        full: Span = (start, end)
        highlight: Span = (start + hl_start, start + hl_end)
        plans.append((full, highlight))
    return plans


def chunk_paragraph(
    document: "Document",
    min_paragraph_tokens: int = DEFAULT_MIN_PARAGRAPH_TOKENS,
    max_paragraph_tokens: int = DEFAULT_MAX_PARAGRAPH_TOKENS,
    chunked_at: str | None = None,
) -> list[Chunk]:
    """Chunk a document by paragraph, merging orphans and highlighting first sentences."""
    plans = _plan_paragraph(document.body, min_paragraph_tokens, max_paragraph_tokens)
    return materialize_chunks(document, plans, chunked_at=chunked_at)
