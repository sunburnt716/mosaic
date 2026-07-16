"""
Tests for processing/text_metrics.py — the shared counting primitives.

These are the measurements inference and validation both rely on, so they are pinned
directly rather than only through their callers.
"""

from __future__ import annotations

from extraction.text_metrics import (
    count_filing_markers,
    count_paragraphs,
    count_tokens,
    paragraph_spans,
    sentence_spans,
)


class TestCountTokens:
    def test_counts_whitespace_delimited_words(self):
        assert count_tokens("one two three") == 3

    def test_collapses_runs_of_whitespace(self):
        assert count_tokens("  one   two\tthree\n four ") == 4

    def test_empty_is_zero(self):
        assert count_tokens("") == 0
        assert count_tokens("   \n\t ") == 0


class TestCountParagraphs:
    def test_blank_line_separated_blocks(self):
        assert count_paragraphs("para one\n\npara two\n\npara three") == 3

    def test_single_block_is_one_paragraph(self):
        assert count_paragraphs("a single unbroken sentence with no breaks") == 1

    def test_single_newline_lines_fall_back_to_line_count(self):
        assert count_paragraphs("line one\nline two\nline three") == 3

    def test_empty_is_zero(self):
        assert count_paragraphs("") == 0
        assert count_paragraphs("   \n  ") == 0


class TestParagraphSpans:
    """paragraph_spans is the offset-returning sibling count_paragraphs is defined on."""

    def test_spans_slice_back_to_blank_line_blocks(self):
        text = "para one\n\npara two\n\npara three"
        assert [text[s:e] for s, e in paragraph_spans(text)] == [
            "para one",
            "para two",
            "para three",
        ]

    def test_spans_are_tight_excluding_surrounding_whitespace(self):
        text = "  para one  \n\n  para two  "
        assert [text[s:e] for s, e in paragraph_spans(text)] == ["para one", "para two"]

    def test_single_newline_lines_each_span(self):
        text = "line one\nline two\nline three"
        assert [text[s:e] for s, e in paragraph_spans(text)] == [
            "line one",
            "line two",
            "line three",
        ]

    def test_empty_has_no_spans(self):
        assert paragraph_spans("") == []
        assert paragraph_spans("   \n  ") == []

    def test_count_paragraphs_equals_span_count(self):
        for text in ("a\n\nb\n\nc", "one solid block", "l1\nl2", "", "   "):
            assert count_paragraphs(text) == len(paragraph_spans(text))


class TestSentenceSpans:
    def test_splits_on_terminators(self):
        text = "First sentence. Second sentence! Third one?"
        assert [text[s:e] for s, e in sentence_spans(text)] == [
            "First sentence.",
            "Second sentence!",
            "Third one?",
        ]

    def test_does_not_split_mid_abbreviation(self):
        # The terminator regex only matches '.'/'!'/'?' followed by whitespace-or-end, so it
        # never splits between "U" and "S" in "U.S." (no whitespace between them). It still
        # (mis)terminates right after "U.S." itself, since that period IS followed by a space
        # — a known, pre-existing limitation this refactor carries forward unchanged, not a
        # true abbreviation-aware sentence boundary.
        text = "The U.S. market rallied."
        assert [text[s:e] for s, e in sentence_spans(text)] == ["The U.S.", "market rallied."]

    def test_trailing_text_with_no_terminator_is_final_sentence(self):
        text = "First sentence. trailing text with no terminator"
        assert [text[s:e] for s, e in sentence_spans(text)] == [
            "First sentence.",
            "trailing text with no terminator",
        ]

    def test_spans_are_tight_excluding_surrounding_whitespace(self):
        text = "  First sentence.   Second sentence.  "
        assert [text[s:e] for s, e in sentence_spans(text)] == [
            "First sentence.",
            "Second sentence.",
        ]

    def test_single_sentence_no_terminator(self):
        text = "no terminator at all   "
        assert [text[s:e] for s, e in sentence_spans(text)] == ["no terminator at all"]

    def test_empty_or_whitespace_only_has_no_spans(self):
        assert sentence_spans("") == []
        assert sentence_spans("   \n\t ") == []


class TestCountFilingMarkers:
    def test_counts_distinct_markers_not_occurrences(self):
        # "Item 1.01" matches two patterns (Item N.N and Item N), "Risk Factors" one;
        # repetition of the same phrase must not inflate the count.
        text = "Item 1.01 ... Risk Factors ... Risk Factors again ... Item 1.01 again"
        assert count_filing_markers(text) == 3

    def test_is_case_insensitive_for_phrase_markers(self):
        assert count_filing_markers("management's discussion and analysis") >= 1

    def test_plain_prose_has_no_markers(self):
        assert count_filing_markers("A normal news sentence about the market today.") == 0
