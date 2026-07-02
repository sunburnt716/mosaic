"""Section-header detection — regex heuristics now, LLM-swappable later.

Structured documents (SEC filings, regulatory text) carry explicit section headers
("RISK FACTORS", "Item 1A."). The section chunker splits on these so each chunk is a
coherent unit rather than an arbitrary token window. This module is the *only* place
that decides what a header looks like; swap the internals for a model later and every
caller is unaffected because the signature is fixed:

  detect_section_headers(text) -> list[(start_char, end_char)]

Each returned span covers the header *line* (not the section body), in document order.
Heuristics, most-specific first:
  - numbered items   "Item 1.", "Item 1A.", "Part II", "1.1 ..."  (EDGAR item structure)
  - all-caps lines   "RISK FACTORS", "MANAGEMENT'S DISCUSSION"     (EDGAR section titles)
  - keyword lines    short heading-like lines naming a known section topic

To avoid mistaking prose for a heading, the keyword and all-caps rules only fire on
short lines (a real header is a label, not a sentence) and never on lines that read as
a sentence (trailing '.'), except the numbered-item rule which legitimately ends in '.'.
"""

import re

# "Item 1.", "Item 1A.", "Part II", "Section 3", leading numeric "1.2 Foo".
_NUMBERED_RE = re.compile(r"(?i)^(item|part|section)\s+\d+[a-z]?\b|^\d+(\.\d+)*\.?\s+\S")

# Common financial-filing section topics; matched only on short, heading-like lines.
_KEYWORD_RE = re.compile(
    r"(?i)\b(risk\s+factors?|management|discussion|financial|business|"
    r"properties|legal\s+proceedings|controls|governance|compensation)\b"
)

# A header is a label, not a paragraph: cap its length so long prose never qualifies.
_MAX_HEADER_LEN = 80


def _looks_like_header(line: str) -> bool:
    """True if a single (stripped) line reads as a section header under the heuristics."""
    if not line:
        return False

    # Numbered items are the strongest signal and may legitimately end in '.'.
    if _NUMBERED_RE.search(line):
        return True

    # Beyond here we require a short, heading-shaped line to avoid matching prose.
    if len(line) > _MAX_HEADER_LEN:
        return False

    letters = [c for c in line if c.isalpha()]
    # All-caps title (has letters, none lowercase): "RISK FACTORS".
    if letters and not any(c.islower() for c in line):
        return True

    # Keyword heading: a short line naming a known section topic and not a full sentence.
    if not line.endswith(".") and _KEYWORD_RE.search(line):
        return True

    return False


def detect_section_headers(text: str) -> list[tuple[int, int]]:
    """Return the (start_char, end_char) span of every line that looks like a header."""
    headers: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if _looks_like_header(stripped):
            start = offset + (len(line) - len(line.lstrip()))
            end = offset + len(line.rstrip())
            headers.append((start, end))
        offset += len(line)
    return headers
