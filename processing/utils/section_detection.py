"""
Section-header detection for the Phase 1 section chunker — regex heuristics now,
LLM-swappable later.

Filings and regulatory text are organized under explicit headers ("RISK FACTORS",
"Item 1A."), and the section chunker splits on these so each chunk is a coherent unit
rather than an arbitrary token window. This module decides what a header line looks like:

  detect_section_headers(text) -> list[(start_char, end_char)]

Each span covers a header *line* (not the section body), in document order. Two signals:
  - filing markers   — reuses `text_metrics.FILING_MARKER_PATTERNS`, the same vocabulary
                       `count_filing_markers` uses for type inference, so "what a filing
                       marker is" is defined in exactly one place and can never drift.
  - all-caps titles  — "UNITED STATES SECURITIES AND EXCHANGE COMMISSION", "RISK FACTORS"

To avoid mistaking prose that merely mentions a marker for a heading, both signals only
fire on short, heading-shaped lines (a real header is a label, not a sentence).

Future extension: swap the internals for model-based detection; the signature is fixed, so
callers are unaffected.
"""

from __future__ import annotations

from processing.text_metrics import FILING_MARKER_PATTERNS

# A header is a label, not a paragraph: cap its length so long prose never qualifies.
_MAX_HEADER_LEN = 80


def _looks_like_header(line: str) -> bool:
    """True if a single (stripped) line reads as a section header under the heuristics."""
    if not line or len(line) > _MAX_HEADER_LEN:
        return False

    # A filing marker appearing on a short line is a section header ("Item 1.01 …").
    if any(pattern.search(line) for pattern in FILING_MARKER_PATTERNS):
        return True

    # All-caps title (has letters, none lowercase): "RISK FACTORS", "FORM 8-K".
    if any(c.isalpha() for c in line) and not any(c.islower() for c in line):
        return True

    return False


def detect_section_headers(text: str) -> list[tuple[int, int]]:
    """Return the (start_char, end_char) span of every line that looks like a header."""
    headers: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if _looks_like_header(line.strip()):
            start = offset + (len(line) - len(line.lstrip()))
            end = offset + len(line.rstrip())
            headers.append((start, end))
        offset += len(line)
    return headers
