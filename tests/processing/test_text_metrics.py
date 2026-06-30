"""
Tests for processing/text_metrics.py — the shared counting primitives.

These are the measurements inference and validation both rely on, so they are pinned
directly rather than only through their callers.
"""

from __future__ import annotations

from processing.text_metrics import count_filing_markers, count_paragraphs, count_tokens


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
