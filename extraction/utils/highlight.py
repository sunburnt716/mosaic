"""
Highlight-span selection — the surgical excerpt inside a chunk.

The dual-span design (see extraction/chunk.py) hands generation two things: the whole
chunk for embedding/context (`full_span`) and a precise excerpt to cite back to the user
(`highlight_span`). This module picks that excerpt. Shared by the paragraph and section
chunkers so the heuristic lives in one place (fixed chunks highlight the whole window and
don't call this).

Phase 1 heuristic: the first sentence of the chunk (optionally past a leading header).
Model-ranked "key sentence" selection is Phase 2+; the signature stays the same when it lands.
Sentence boundaries come from `text_metrics.sentence_spans` — the same definition the
Generation Pipeline's Phase 5 citation sentence-selection uses — so the two can never disagree
about what counts as a sentence.

  select_highlight_span(text, start=0) -> (start_char, end_char)   relative to `text`
"""

from __future__ import annotations

from extraction.text_metrics import sentence_spans


def select_highlight_span(text: str, start: int = 0) -> tuple[int, int]:
    """Return the span of the first sentence in `text` at or after `start`.

    `start` lets a caller skip a leading header line (the section chunker passes the offset
    just past the header) — sentences are located fresh within `text[start:]`, so text before
    `start` (e.g. a header with no terminator of its own) never bleeds into the result. Falls
    back to `(start, len(text))` when there is no real content at or after `start`.
    """
    spans = sentence_spans(text[start:])
    if not spans:
        return (start, len(text))
    first_start, first_end = spans[0]
    return (start + first_start, start + first_end)
