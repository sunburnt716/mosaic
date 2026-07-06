"""
Pure text-measurement helpers shared across the processing layer.

These functions answer one question only — "how big and how structured is this
text?" — with no knowledge of document types or policy thresholds. They are the
raw measurements that both type inference and validation build their decisions on
top of. Keeping them here means the counting logic lives in exactly one place, so
inference and validation can never drift apart on what a "token", "paragraph", or
"filing marker" is.

Phase 1 chunking builds on the same primitives: it splits articles on the paragraph
boundaries defined here (`paragraph_spans`, the offset-returning sibling of
`count_paragraphs`) and locates filing section headers with the same marker patterns
`count_filing_markers` uses (`FILING_MARKER_PATTERNS`) — so structure is defined once
and inference, validation, and chunking can never disagree about it.

Everything here is pure: deterministic, side-effect-free, no I/O, and — per the
Phase 0 hard constraint — no model of any kind. `count_tokens` is a deliberately
cheap whitespace proxy, not a real tokenizer; heuristics do not need that precision.
(Phase 1 chunk *sizing* uses the real MiniLM tokenizer in `processing/utils/
tokenization.py`; that is a distinct, embedding-aligned notion of "token" and is kept
separate on purpose — see that module.)
"""

from __future__ import annotations

import re

# Structural markers that are highly characteristic of SEC filings. Each pattern is
# distinctive enough that a handful of them appearing together is a strong signal of
# a filing, while a single stray mention in an article is not (callers require more
# than one distinct marker — see type_inference.FILING_MARKER_MIN). These double as
# the section-header vocabulary the Phase 1 section chunker splits on.
#
# Patterns are matched case-insensitively. Listed roughly from most to least specific.
FILING_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bItem\s+\d+\.\d+\b", re.IGNORECASE),  # "Item 1.01", "Item 2.02"
    re.compile(r"\bItem\s+\d+[A-Z]?\b", re.IGNORECASE),  # "Item 1A", "Item 7"
    re.compile(r"\bRisk\s+Factors\b", re.IGNORECASE),
    re.compile(r"\bManagement[’']?s\s+Discussion\s+and\s+Analysis\b", re.IGNORECASE),
    re.compile(r"\bMD&A\b", re.IGNORECASE),
    re.compile(r"\bForward[-\s]Looking\s+Statements\b", re.IGNORECASE),
    re.compile(r"\bTable\s+of\s+Contents\b", re.IGNORECASE),
    re.compile(r"\bPART\s+[IVX]+\b"),  # "PART I" (uppercase only)
)

# Paragraphs are normally separated by a blank line. Some sources separate them with a
# single newline instead, so we fall back to that when no blank-line boundary is found.
_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def count_tokens(text: str) -> int:
    """Return an approximate token count: the number of whitespace-delimited words.

    A cheap, deterministic proxy for length — good enough for the size thresholds the
    heuristics use, and intentionally free of any tokenizer or model dependency.
    """
    return len(text.split())


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) inward past leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Return the (start_char, end_char) span of each paragraph block in `text`.

    The offset-returning sibling of `count_paragraphs`: same boundary definition
    (blank-line blocks, falling back to non-empty lines when there is no blank-line
    boundary), but returns tight character spans so the paragraph chunker can slice
    the exact text. Empty/whitespace-only text has no paragraphs. `count_paragraphs`
    is defined as the length of this, so the two can never drift.
    """
    if not text.strip():
        return []

    # Blank-line-separated blocks, tracked in original-text coordinates.
    block_spans: list[tuple[int, int]] = []
    pos = 0
    for match in _BLANK_LINE_RE.finditer(text):
        block_spans.append((pos, match.start()))
        pos = match.end()
    block_spans.append((pos, len(text)))

    blocks = [(s, e) for s, e in (_trim_span(text, s, e) for s, e in block_spans) if s < e]
    if len(blocks) > 1:
        return blocks

    # No blank-line boundary: treat each non-empty line as a paragraph.
    line_spans: list[tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        s, e = _trim_span(text, pos, pos + len(line))
        if s < e:
            line_spans.append((s, e))
        pos += len(line)
    return line_spans


def count_paragraphs(text: str) -> int:
    """Return the number of paragraph-like blocks in the text.

    A blank line is the canonical paragraph boundary. When the text has no blank-line
    boundaries (some feeds emit single-newline-separated paragraphs, or one solid
    block), fall back to counting non-empty lines so a single blob still reads as one
    paragraph. Empty/whitespace-only text has zero paragraphs. Defined as the count of
    `paragraph_spans` so the two share one boundary definition.
    """
    return len(paragraph_spans(text))


def count_filing_markers(text: str) -> int:
    """Return how many DISTINCT filing-marker patterns appear in the text.

    Counts distinct patterns (not total occurrences) so the signal reflects
    structural variety — a real filing hits several different markers, whereas an
    article that merely quotes one phrase like "risk factors" hits only one.
    """
    return sum(1 for pattern in FILING_MARKER_PATTERNS if pattern.search(text))
