"""
Pure text-measurement helpers shared across the processing layer.

These functions answer one question only — "how big and how structured is this
text?" — with no knowledge of document types or policy thresholds. They are the
raw measurements that both type inference and validation build their decisions on
top of. Keeping them here means the counting logic lives in exactly one place, so
inference and validation can never drift apart on what a "token", "paragraph", or
"filing marker" is.

Everything here is pure: deterministic, side-effect-free, no I/O, and — per the
Phase 0 hard constraint — no model of any kind. Token counting is a deliberately
cheap whitespace proxy, not a real tokenizer; heuristics do not need that precision.
"""

from __future__ import annotations

import re

# Structural markers that are highly characteristic of SEC filings. Each pattern is
# distinctive enough that a handful of them appearing together is a strong signal of
# a filing, while a single stray mention in an article is not (callers require more
# than one distinct marker — see type_inference.FILING_MARKER_MIN).
#
# Patterns are matched case-insensitively. Listed roughly from most to least specific.
_FILING_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
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


def count_paragraphs(text: str) -> int:
    """Return the number of paragraph-like blocks in the text.

    A blank line is the canonical paragraph boundary. When the text has no blank-line
    boundaries (some feeds emit single-newline-separated paragraphs, or one solid
    block), fall back to counting non-empty lines so a single blob still reads as one
    paragraph. Empty/whitespace-only text has zero paragraphs.
    """
    stripped = text.strip()
    if not stripped:
        return 0

    blocks = [block for block in _BLANK_LINE_RE.split(stripped) if block.strip()]
    if len(blocks) > 1:
        return len(blocks)

    # No blank-line boundary: treat each non-empty line as a paragraph, but never
    # report zero for non-empty text.
    lines = [line for line in stripped.splitlines() if line.strip()]
    return len(lines) if lines else 1


def count_filing_markers(text: str) -> int:
    """Return how many DISTINCT filing-marker patterns appear in the text.

    Counts distinct patterns (not total occurrences) so the signal reflects
    structural variety — a real filing hits several different markers, whereas an
    article that merely quotes one phrase like "risk factors" hits only one.
    """
    return sum(1 for pattern in _FILING_MARKER_PATTERNS if pattern.search(text))
