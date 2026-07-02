"""
Highlight-span selection — the surgical excerpt inside a chunk.

The dual-span design (see processing/chunk.py) hands generation two things: the whole
chunk for embedding/context (`full_span`) and a precise excerpt to cite back to the user
(`highlight_span`). This module picks that excerpt. Shared by the paragraph and section
chunkers so the heuristic lives in one place (fixed chunks highlight the whole window and
don't call this).

Phase 1 heuristic: the first sentence of the chunk (optionally past a leading header).
Model-ranked "key sentence" selection is Phase 2+; the signature stays the same when it lands.

  select_highlight_span(text, start=0) -> (start_char, end_char)   relative to `text`
"""

from __future__ import annotations

import re

# Sentence terminator followed by whitespace or end-of-text (avoids splitting "U.S.").
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def select_highlight_span(text: str, start: int = 0) -> tuple[int, int]:
    """Return the span of the first sentence in `text` at or after `start`.

    `start` lets a caller skip a leading header line (the section chunker passes the offset
    just past the header). Falls back to the whole trimmed remainder when the text has no
    sentence terminator.
    """
    # Advance past leading whitespace so the highlight begins on real content.
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return (0, len(text))

    match = _SENTENCE_END_RE.search(text, start)
    end = match.end() if match else len(text.rstrip())
    return (start, end)
