"""
Contract tests for processing/utils/highlight.py.

The highlight span is the excerpt generation cites back to the user. Phase 1 heuristic: the
first sentence, optionally past a leading header offset. Pins that behaviour and the
whole-remainder fallback when no sentence terminator exists.
"""

from __future__ import annotations

from processing.utils.highlight import select_highlight_span


class TestSelectHighlightSpan:
    def test_first_sentence(self):
        text = "First sentence here. Second sentence follows."
        start, end = select_highlight_span(text)
        assert text[start:end] == "First sentence here."

    def test_skips_leading_whitespace(self):
        text = "   Leading spaces then text. More."
        start, end = select_highlight_span(text)
        assert text[start:end] == "Leading spaces then text."

    def test_start_offset_skips_header(self):
        text = "RISK FACTORS\nThe first real sentence. Then more."
        start, end = select_highlight_span(text, start=text.index("\n"))
        assert text[start:end] == "The first real sentence."

    def test_no_terminator_returns_trimmed_remainder(self):
        text = "no terminator at all   "
        start, end = select_highlight_span(text)
        assert text[start:end] == "no terminator at all"

    def test_all_whitespace_is_safe(self):
        start, end = select_highlight_span("     ")
        assert start <= end
